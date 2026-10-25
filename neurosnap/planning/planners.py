"""Common planner interface used by the mission runner and the benchmark.

plan(P, v0, a0, t0) -> trajectory with .state(t) -> (p, v, a, j), .t0, .t_end
P[0] is the current (reference) position, P[1:] the remaining gates and the finish.
"""
import time

import numpy as np

from ..config import TrajLimits
from ..traj.minsnap import plan_minsnap


class LinearTraj:
    """Straight lines between waypoints with a trapezoidal speed profile along the path."""

    def __init__(self, P, v_start, v_cruise, acc, t0=0.0):
        self.P = np.asarray(P, float)
        seg = np.diff(self.P, axis=0)
        self.len = np.linalg.norm(seg, axis=1)
        self.dirs = seg / np.maximum(self.len[:, None], 1e-9)
        self.cum = np.concatenate([[0.0], np.cumsum(self.len)])
        L = self.cum[-1]
        v0 = min(v_start, v_cruise)
        self.v0, self.vc, self.acc = v0, v_cruise, acc
        d_up = (v_cruise ** 2 - v0 ** 2) / (2 * acc)
        d_dn = v_cruise ** 2 / (2 * acc)
        if d_up + d_dn > L:                        # triangular profile
            vp = np.sqrt(max((2 * acc * L + v0 ** 2) / 2, v0 ** 2))
            d_up = (vp ** 2 - v0 ** 2) / (2 * acc)
            d_dn = vp ** 2 / (2 * acc)
            self.vc = vp
        self.t_up = (self.vc - v0) / acc
        self.t_cr = max(L - d_up - d_dn, 0.0) / self.vc
        self.t_dn = self.vc / acc
        self.d_up = d_up
        self.t0 = t0
        self.L = L

    @property
    def duration(self):
        return self.t_up + self.t_cr + self.t_dn

    @property
    def t_end(self):
        return self.t0 + self.duration

    def _s(self, t):
        t = np.clip(t - self.t0, 0, self.duration)
        if t < self.t_up:
            return self.v0 * t + 0.5 * self.acc * t * t, self.v0 + self.acc * t, self.acc
        t -= self.t_up
        if t < self.t_cr:
            return self.d_up + self.vc * t, self.vc, 0.0
        t -= self.t_cr
        return (self.d_up + self.vc * self.t_cr + self.vc * t - 0.5 * self.acc * t * t,
                self.vc - self.acc * t, -self.acc)

    def state(self, t):
        s, sd, sdd = self._s(t)
        i = int(np.clip(np.searchsorted(self.cum, s, side='right') - 1, 0, len(self.len) - 1))
        p = self.P[i] + self.dirs[i] * (s - self.cum[i])
        return p, self.dirs[i] * sd, self.dirs[i] * sdd, np.zeros(3)

    def sample(self, dt=0.02):
        ts = np.arange(self.t0, self.t_end + 1e-9, dt)
        st = [self.state(t) for t in ts]
        return ts, np.array([s[0] for s in st]), np.array([s[1] for s in st]), np.array([s[2] for s in st])


class Planner:
    name = 'base'

    def __init__(self):
        self.times_ms = []

    def plan(self, P, v0, a0, t0=0.0):
        t = time.perf_counter()
        traj = self._plan(np.asarray(P, float), np.asarray(v0, float), np.asarray(a0, float), t0)
        self.times_ms.append((time.perf_counter() - t) * 1e3)
        return traj


class StraightLinePlanner(Planner):
    name = 'straight_line'

    def __init__(self, lim=None, v_cruise=1.8, acc=3.0, name=None):
        super().__init__()
        self.name = name or self.name
        self.lim = lim or TrajLimits()
        self.v_cruise, self.acc = v_cruise, acc

    def _plan(self, P, v0, a0, t0):
        return LinearTraj(P, float(np.linalg.norm(v0)), self.v_cruise, self.acc, t0)


class MinSnapPlanner(Planner):
    """Classical minimum-snap QP; optimal_times=True adds Mellinger-style time allocation."""

    def __init__(self, lim=None, optimal_times=True):
        super().__init__()
        self.lim = lim or TrajLimits()
        self.optimal_times = optimal_times
        self.name = 'minsnap_opt_T' if optimal_times else 'minsnap_heur_T'

    def _plan(self, P, v0, a0, t0):
        return plan_minsnap(P, self.lim, v0, a0, self.optimal_times, t0)


class LearnedPlanner(Planner):
    def __init__(self, weights, mode='full', lim=None, name=None):
        super().__init__()
        from .neural import NeuralPlanner
        import torch
        torch.set_num_threads(1)
        self.lim = lim or TrajLimits()
        self.net = NeuralPlanner(weights, mode, self.lim)
        self.name = name or f'neural_{mode}'

    def _plan(self, P, v0, a0, t0):
        return self.net.plan(P, v0, a0, t0)
