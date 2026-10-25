"""Planner dataset: random (re-)planning problems labelled by the minimum-snap teacher.

A problem is: current state (position, velocity, acceleration) + the remaining
waypoints (gate centres and the finish point). Half of the problems start at rest
on a fresh course; the other half are taken mid-flight from a teacher trajectory
(with perturbed velocity/acceleration) so that the network learns the receding-
horizon re-planning distribution it sees in closed loop.

All problems are stored in a canonical frame: origin at the current position and
x-axis pointing horizontally towards the first waypoint.
"""
import os
from multiprocessing import Pool

import numpy as np

from ..config import TrajLimits
from ..geometry import rot_z
from ..world.course import sample_course
from .minsnap import optimize_times, solve_fixed_times
from .poly import PiecewisePoly, enforce_limits

MAX_SEG = 8
MAX_KNOTS = MAX_SEG + 1


def canonical_frame(P):
    """Translation and rotation that map P[0] to the origin and P[1] onto the +x half-plane."""
    d = P[1] - P[0]
    yaw = np.arctan2(d[1], d[0])
    return P[0].copy(), rot_z(yaw)


def to_canonical(P, v0, a0):
    o, R = canonical_frame(P)
    return (P - o) @ R, v0 @ R, a0 @ R, (o, R)


def teacher(P, v0, a0, lim):
    """Minimum-snap with optimised times, made feasible by stretching + re-solving."""
    T, _ = optimize_times(P, lim, v0, a0)

    def resolve(Tn):
        return PiecewisePoly(Tn, solve_fixed_times(P, Tn, v0, a0)[0])

    traj, scale = enforce_limits(resolve(T), lim, resolve)
    return traj, scale


def _sample_problem(rng, lim):
    course = sample_course(rng, with_obstacles=False)
    P = np.vstack([course.start, course.waypoints()])
    v0 = np.zeros(3)
    a0 = np.zeros(3)
    if rng.random() < 0.5:
        full, _ = teacher(P, v0, a0, lim)
        t = rng.uniform(0.0, 0.92) * full.duration
        p, v, a, _ = full.state(t)
        v = v + rng.normal(0, 0.25, 3)
        a = a + rng.normal(0, 0.8, 3)
        nxt = int(np.searchsorted(full.tk, t + 0.15, side='left'))
        nxt = min(max(nxt, 1), len(P) - 1)
        P = np.vstack([p, P[nxt:]])
        v0, a0 = v, a
    if len(P) - 1 > MAX_SEG:
        P = P[:MAX_SEG + 1]
    if np.linalg.norm(P[1, :2] - P[0, :2]) < 0.05:
        return None
    Pc, vc, ac, _ = to_canonical(P, v0, a0)
    try:
        traj, scale = teacher(Pc, vc, ac, lim)
    except np.linalg.LinAlgError:
        return None
    if scale > 3.0 or not np.all(np.isfinite(traj.knots)):
        return None
    return Pc, vc, ac, traj


def _worker(args):
    seed, count = args
    rng = np.random.default_rng(seed)
    lim = TrajLimits()
    n = 0
    out = dict(P=[], v0=[], a0=[], T=[], X=[], nseg=[], J=[])
    while n < count:
        s = _sample_problem(rng, lim)
        if s is None:
            continue
        Pc, vc, ac, traj = s
        k = len(Pc)
        Pp = np.zeros((MAX_KNOTS, 3))
        Pp[:k] = Pc
        Tp = np.ones(MAX_SEG)
        Tp[:k - 1] = traj.T
        Xp = np.zeros((MAX_KNOTS, 4, 3))
        Xp[:k] = traj.knots
        out['P'].append(Pp)
        out['v0'].append(vc)
        out['a0'].append(ac)
        out['T'].append(Tp)
        out['X'].append(Xp)
        out['nseg'].append(k - 1)
        out['J'].append(traj.snap_cost())
        n += 1
    return {k: np.array(v) for k, v in out.items()}


def generate(n_samples, path, seed=0, workers=6, chunk=500):
    jobs = [(seed * 100003 + i, chunk) for i in range(int(np.ceil(n_samples / chunk)))]
    parts = []
    with Pool(workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_worker, jobs)):
            parts.append(part)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f'  teacher-labelled {min((i + 1) * chunk, n_samples)}/{n_samples}', flush=True)
    data = {k: np.concatenate([p[k] for p in parts])[:n_samples] for k in parts[0]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)
    return data
