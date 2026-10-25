"""Cascaded PID trajectory tracker (position/velocity -> attitude -> body rate -> mixer).

The reference paper tracks its trajectories with a hand-designed nonlinear controller;
here a standard PX4/DSL-style cascade is used instead, with acceleration feed-forward
from the trajectory and an optional learned residual model that cancels the rotor
drag the cascade does not model.
"""
import numpy as np

from ..config import G, ControlParams
from ..geometry import vee


class CascadedPID:
    def __init__(self, quad, cp=None, residual=None):
        self.quad = quad
        self.cp = cp or ControlParams()
        self.residual = residual            # callable(state dict) -> world-frame acc, or None
        self.kp = np.array(self.cp.kp_pos)
        self.ki = np.array(self.cp.ki_pos)
        self.kd = np.array(self.cp.kd_pos)
        self.kp_att = np.array(self.cp.kp_att)
        self.kp_rate = np.array(self.cp.kp_rate)
        self.ki_rate = np.array(self.cp.ki_rate)
        self.reset()

    def reset(self):
        self.i_pos = np.zeros(3)
        self.i_rate = np.zeros(3)
        self.last_res = np.zeros(3)

    def compute(self, p, v, R, w, ref, dt):
        """ref = dict(p, v, a, yaw). Returns motor RPM commands."""
        qd = self.quad.qp
        cp = self.cp
        e_p = ref['p'] - p
        e_v = ref['v'] - v
        # conditional integration (anti-windup): only integrate small tracking errors
        e_i = np.where(np.abs(e_p) < 0.25, e_p, 0.0)
        self.i_pos = np.clip(self.i_pos + self.ki * e_i * dt, -cp.i_limit, cp.i_limit)
        a_cmd = ref['a'] + self.kp * e_p + self.kd * e_v + self.i_pos
        if self.residual is not None:
            self.last_res = self.residual(v, R, self.quad.rpm)
            a_cmd = a_cmd - self.last_res
        f_vec = qd.mass * (a_cmd + np.array([0, 0, G]))
        # tilt limit
        max_tilt = np.radians(cp.max_tilt_deg)
        f_vec[2] = max(f_vec[2], 0.3 * qd.mass * G)
        horiz = np.linalg.norm(f_vec[:2])
        if horiz > f_vec[2] * np.tan(max_tilt):
            f_vec[:2] *= f_vec[2] * np.tan(max_tilt) / horiz
        b3 = f_vec / np.linalg.norm(f_vec)
        b1c = np.array([np.cos(ref['yaw']), np.sin(ref['yaw']), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        thrust = float(np.dot(f_vec, R[:, 2]))
        if 'Rd' in ref:                     # attitude-level command (used by the stability analysis)
            Rd = ref['Rd']
            thrust = qd.mass * G / max(R[2, 2], 0.5)
        # attitude (SO(3) error) -> desired body rates
        e_R = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        w_d = np.clip(-self.kp_att * e_R, -cp.max_rate, cp.max_rate)
        e_w = w_d - w
        self.i_rate = np.clip(self.i_rate + self.ki_rate * e_w * dt, -2.0, 2.0)
        alpha = self.kp_rate * e_w + self.i_rate
        J = self.quad.J
        torque = J @ alpha + np.cross(w, J @ w)
        return self.quad.wrench_to_rpm(thrust, torque)
