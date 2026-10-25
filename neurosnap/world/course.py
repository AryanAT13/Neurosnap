"""Randomised gate courses, the noisy prior map and gate-passing / collision checks."""
from dataclasses import dataclass, field

import numpy as np

from ..config import CourseParams, GateParams
from ..geometry import rot_z


@dataclass
class Gate:
    center: np.ndarray            # (3,) world position of the opening centre
    yaw: float                    # gate normal points along (cos yaw, sin yaw, 0)

    @property
    def R(self):
        """Gate frame: x = normal (flight direction), y = left, z = up."""
        return rot_z(self.yaw)

    @property
    def normal(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    def inner_corners(self, size):
        """Opening corners in world frame, ordered TL, TR, BR, BL as seen when flying through."""
        h = size / 2
        local = np.array([[0, h, h], [0, -h, h], [0, -h, -h], [0, h, -h]], dtype=float)
        return self.center + local @ self.R.T


@dataclass
class Obstacle:
    """Visual distractor (box / pillar) standing on the ground."""
    center: np.ndarray
    size: np.ndarray
    color: tuple


@dataclass
class Course:
    gates: list
    start: np.ndarray
    finish: np.ndarray
    obstacles: list = field(default_factory=list)
    seed: int = 0

    def waypoints(self, gates=None):
        gates = self.gates if gates is None else gates
        return np.array([g.center for g in gates] + [self.finish])


def _dist_point_polyline(p, pts):
    best = np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        best = min(best, np.linalg.norm(p - (a + t * ab)))
    return best


def sample_course(rng, cp=None, gp=None, n_gates=None, with_obstacles=True):
    cp = cp or CourseParams()
    gp = gp or GateParams()
    for _ in range(200):
        n = n_gates or int(rng.integers(cp.n_gates[0], cp.n_gates[1] + 1))
        start = np.array(cp.start, dtype=float)
        pos, heading = start.copy(), 0.0
        gates, ok = [], True
        for i in range(n):
            dh = np.radians(cp.heading_change_deg) * (0.35 if i == 0 else 1.0)
            heading += rng.uniform(-dh, dh)
            d = rng.uniform(*cp.spacing)
            z = np.clip(pos[2] + rng.uniform(-0.35, 0.35) * d, *cp.height)
            c = np.array([pos[0] + d * np.cos(heading), pos[1] + d * np.sin(heading), z])
            yaw = heading + np.radians(rng.uniform(-cp.gate_yaw_jitter_deg, cp.gate_yaw_jitter_deg))
            if any(np.linalg.norm(c[:2] - g.center[:2]) < 2.0 for g in gates) or np.linalg.norm(c[:2] - start[:2]) < 2.0:
                ok = False
                break
            gates.append(Gate(c, yaw))
            pos = c
        if not ok:
            continue
        finish = pos + cp.finish_dist * np.array([np.cos(heading), np.sin(heading), 0.0])
        # the straight path between consecutive gates must not cut through another gate
        pts = np.array([start] + [g.center for g in gates] + [finish])
        clash = False
        for gi, g in enumerate(gates):
            others = [k for k in range(len(pts) - 1) if k not in (gi, gi + 1)]
            for k in others:
                if _dist_point_polyline(g.center, pts[k:k + 2]) < 1.2:
                    clash = True
        if clash:
            continue
        obstacles = []
        if with_obstacles:
            lo, hi = pts.min(0) - 5, pts.max(0) + 5
            for _ in range(int(rng.integers(4, 12))):
                for _ in range(20):
                    c = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]), 0.0])
                    if _dist_point_polyline(c, pts) > 2.2:
                        break
                else:
                    continue
                pillar = rng.random() < 0.5
                size = (np.array([rng.uniform(0.2, 0.5)] * 2 + [rng.uniform(1.5, 3.5)]) if pillar
                        else rng.uniform([0.4, 0.4, 0.3], [1.5, 1.5, 1.2]))
                c[2] = size[2] / 2
                col = tuple(int(v) for v in rng.integers(40, 230, 3))
                obstacles.append(Obstacle(c, size, col))
        return Course(gates, start, finish, obstacles)
    raise RuntimeError('could not sample a valid course')


def noisy_prior(course, rng, sigma=None, yaw_sigma_deg=10.0):
    """The approximate track map the drone receives before take-off."""
    sigma = np.array(sigma if sigma is not None else CourseParams().prior_sigma)
    return [Gate(g.center + rng.normal(0, sigma), g.yaw + np.radians(rng.normal(0, yaw_sigma_deg)))
            for g in course.gates]


POLE_WIDTH = 0.06


def gate_crossing(p0, p1, gate, gp, radius):
    """Classify the motion p0 -> p1 against one gate: None, 'pass' or 'crash'."""
    n = gate.normal
    s0, s1 = np.dot(p0 - gate.center, n), np.dot(p1 - gate.center, n)
    if s0 * s1 > 0 or s0 == s1:
        return None
    x = p0 + (p1 - p0) * (s0 / (s0 - s1))
    local = gate.R.T @ (x - gate.center)
    y, z = abs(local[1]), local[2]
    hi, ho = gp.inner / 2, gp.outer / 2
    if y < hi - radius and abs(z) < hi - radius:
        return 'pass'
    if y < ho + radius and abs(z) < ho + radius:
        return 'crash'
    # vertical stand from the bottom bar to the ground
    if y < POLE_WIDTH / 2 + radius and z < -ho and x[2] > 0:
        return 'crash'
    return None
