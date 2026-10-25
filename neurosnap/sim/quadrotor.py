"""6-DOF rigid-body quadrotor (Crazyflie 2.x, X configuration).

Newton-Euler dynamics with first-order motor lag, rotor drag proportional to the
summed rotor speed (the drag model of gym-pybullet-drones), an optional wind
field and semi-implicit Euler integration at `sim_hz`.
"""
import numpy as np

from ..config import G, QuadParams
from ..geometry import quat_mul, quat_to_rot, quat_from_euler

RPM2RADS = 2 * np.pi / 60


class Quadrotor:
    def __init__(self, qp=None, wind=None):
        self.qp = qp or QuadParams()
        q = self.qp
        a = q.arm / np.sqrt(2)
        # motors: front-right, back-right, back-left, front-left (x forward, y left)
        self.rpos = np.array([[a, -a], [-a, -a], [-a, a], [a, a]])
        self.spin = np.array([-1.0, 1.0, -1.0, 1.0])
        self.mix = np.vstack([np.ones(4), self.rpos[:, 1], -self.rpos[:, 0], self.spin * q.km / q.kf])
        self.mix_inv = np.linalg.inv(self.mix)
        self.J = q.J
        self.J_inv = np.linalg.inv(q.J)
        self.wind = np.zeros(3) if wind is None else np.asarray(wind, float)
        self.ext_force = np.zeros(3)          # external push (gust test), world frame
        self.reset(np.zeros(3))

    def reset(self, pos, yaw=0.0, vel=None):
        self.p = np.asarray(pos, float).copy()
        self.v = np.zeros(3) if vel is None else np.asarray(vel, float).copy()
        self.q = quat_from_euler(0.0, 0.0, yaw)
        self.w = np.zeros(3)
        self.rpm = np.full(4, self.qp.hover_rpm)
        self.acc = np.zeros(3)
        self.t = 0.0

    @property
    def R(self):
        return quat_to_rot(self.q)

    def thrust_forces(self, rpm=None):
        rpm = self.rpm if rpm is None else rpm
        return self.qp.kf * rpm ** 2

    def drag_force(self, R=None):
        """Rotor drag in world frame (the 'unmodelled' aerodynamics the residual model learns)."""
        R = self.R if R is None else R
        omega_sum = np.sum(self.rpm) * RPM2RADS
        coeff = np.array([self.qp.drag_xy, self.qp.drag_xy, self.qp.drag_z])
        v_body = R.T @ (self.v - self.wind)
        return R @ (-coeff * omega_sum * v_body)

    def wrench_to_rpm(self, thrust, torque):
        f = self.mix_inv @ np.concatenate([[thrust], torque])
        return np.clip(np.sqrt(np.maximum(f, 0.0) / self.qp.kf), 0.0, self.qp.max_rpm)

    def step(self, rpm_cmd, dt):
        q = self.qp
        rpm_cmd = np.clip(rpm_cmd, 0.0, q.max_rpm)
        self.rpm += (rpm_cmd - self.rpm) * (1 - np.exp(-dt / q.motor_tau))
        f = self.thrust_forces()
        R = self.R
        F = R @ np.array([0.0, 0.0, f.sum()]) + self.drag_force(R) + self.ext_force - np.array([0, 0, q.mass * G])
        tau = np.array([np.dot(self.rpos[:, 1], f), -np.dot(self.rpos[:, 0], f),
                        np.dot(self.spin, q.km * self.rpm ** 2)])
        self.acc = F / q.mass
        self.v = self.v + self.acc * dt
        self.p = self.p + self.v * dt
        wdot = self.J_inv @ (tau - np.cross(self.w, self.J @ self.w))
        self.w = self.w + wdot * dt
        self.q = self.q + 0.5 * quat_mul(self.q, np.concatenate([[0.0], self.w])) * dt
        self.q /= np.linalg.norm(self.q)
        self.t += dt
        # ground contact
        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[:] = 0.0
