"""Central parameter set for the whole stack.

Every physical constant lives here so the report and the code cite one table.
The airframe follows the Bitcraze Crazyflie 2.x "CF2X" model used by
gym-pybullet-drones (mass, inertia, kf/km, thrust-to-weight, drag coefficients).
"""
import os
from dataclasses import dataclass, field

import numpy as np

G = 9.81
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT, 'models')
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'docs', 'figures')
# planner checkpoint deployed in the mission / benchmark (selected on held-out end-to-end snap cost)
PLANNER_TAG = 'dagger'


@dataclass
class QuadParams:
    """Crazyflie 2.x in X configuration (values from gym-pybullet-drones cf2x.urdf)."""
    mass: float = 0.027                       # kg
    arm: float = 0.0397                       # m, centre to motor
    inertia: tuple = (1.4e-5, 1.4e-5, 2.17e-5)  # kg m^2
    kf: float = 3.16e-10                      # N / RPM^2
    km: float = 7.94e-12                      # N m / RPM^2
    thrust2weight: float = 2.25
    motor_tau: float = 0.02                   # s, first-order rotor-speed lag
    # rotor drag: F = -drag * sum(omega_i) * v_body   (omega in rad/s)
    drag_xy: float = 9.1785e-7
    drag_z: float = 10.311e-7
    radius: float = 0.06                      # m, collision sphere

    @property
    def J(self):
        return np.diag(self.inertia)

    @property
    def hover_rpm(self):
        return np.sqrt(self.mass * G / (4 * self.kf))

    @property
    def max_rpm(self):
        return np.sqrt(self.thrust2weight * self.mass * G / (4 * self.kf))


@dataclass
class ControlParams:
    """Cascaded PID gains (position -> velocity -> attitude -> body rate)."""
    rate_hz: float = 500.0
    kp_pos: tuple = (6.0, 6.0, 8.0)
    ki_pos: tuple = (0.5, 0.5, 1.0)
    kd_pos: tuple = (4.0, 4.0, 5.0)
    i_limit: float = 1.5                      # m/s^2, integrator clamp
    kp_att: tuple = (14.0, 14.0, 6.0)         # 1/s
    kp_rate: tuple = (60.0, 60.0, 20.0)       # 1/s
    ki_rate: tuple = (20.0, 20.0, 5.0)
    max_tilt_deg: float = 45.0
    max_rate: float = 12.0                    # rad/s


@dataclass
class CameraParams:
    """Forward-facing FPV camera, tilted up to compensate pitch in forward flight."""
    width: int = 192
    height: int = 144
    hfov_deg: float = 90.0
    tilt_up_deg: float = 15.0
    offset: tuple = (0.03, 0.0, 0.01)         # m, camera position in body frame
    rate_hz: float = 15.0
    max_range: float = 12.0

    @property
    def fx(self):
        return (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)

    @property
    def K(self):
        f = self.fx
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])


@dataclass
class GateParams:
    inner: float = 0.8                        # m, side of the square opening
    border: float = 0.15                      # m, width of the frame bars
    depth: float = 0.05                       # m, thickness of the frame

    @property
    def outer(self):
        return self.inner + 2 * self.border


@dataclass
class TrajLimits:
    """Dynamic limits shared by the teacher, the neural planner and the safety layer."""
    v_max: float = 3.0                        # m/s
    a_max: float = 6.0                        # m/s^2 (thrust headroom: 2.25 g total)
    rho: float = 40.0                         # time penalty in J_snap + rho * sum(T)
    v_cruise: float = 1.8                     # heuristic time-allocation speed


@dataclass
class CourseParams:
    n_gates: tuple = (3, 7)                   # inclusive range
    spacing: tuple = (2.5, 5.0)               # m between gates
    heading_change_deg: float = 60.0
    height: tuple = (0.8, 2.0)
    gate_yaw_jitter_deg: float = 20.0
    finish_dist: float = 2.0
    start: tuple = (0.0, 0.0, 1.0)
    # prior map handed to the drone: true gate centre + N(0, sigma)
    prior_sigma: tuple = (0.45, 0.45, 0.2)


@dataclass
class Config:
    quad: QuadParams = field(default_factory=QuadParams)
    ctrl: ControlParams = field(default_factory=ControlParams)
    cam: CameraParams = field(default_factory=CameraParams)
    gate: GateParams = field(default_factory=GateParams)
    lim: TrajLimits = field(default_factory=TrajLimits)
    course: CourseParams = field(default_factory=CourseParams)
    sim_hz: float = 1000.0
    replan_hz: float = 10.0
    max_mission_time: float = 40.0
