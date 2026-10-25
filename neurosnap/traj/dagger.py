"""Receding-horizon planner rollouts and DAgger data collection.

The network is trained on states that lie on *teacher* trajectories, but in flight it
re-plans every 0.1 s from states on its *own* previous plan. Small biases then compound
(classic imitation-learning covariate shift): each re-plan is a little slower than the
last and the vehicle keeps decelerating. DAgger (Ross et al., AISTATS 2011) fixes this by
rolling out the learned planner, labelling the states it actually visits with the
teacher, and retraining on the aggregate.

A rollout here is planner-only (no physics, no perception): plan, advance the reference
by dt, re-plan from the reference state, exactly as `Mission` does with perfect tracking.
"""
from multiprocessing import Pool

import numpy as np

from ..config import TrajLimits
from ..world.course import sample_course
from .dataset import MAX_KNOTS, MAX_SEG, teacher, to_canonical

COMMIT_DIST = 0.7


def receding_rollout(planner, P, dt=0.1, max_time=60.0, collect=False):
    """Fly the reference only. Returns (time to finish, initial plan duration, visited problems)."""
    P = np.asarray(P, float)
    traj = planner.plan(P, np.zeros(3), np.zeros(3), 0.0)
    first_duration = traj.duration
    wps = list(P[1:])
    t = 0.0
    visited = []
    while t < max_time:
        t += dt
        if t >= traj.t_end:
            break
        # the next waypoint is reached when the current plan passes its first knot
        while len(wps) > 1 and hasattr(traj, 'tk') and t >= traj.t0 + traj.tk[1] - 1e-9:
            wps.pop(0)
            p, v, a, _ = traj.state(t)
            traj = planner.plan(np.vstack([p, wps]), v, a, t)
        p, v, a, _ = traj.state(t)
        if np.linalg.norm(wps[0] - p) <= COMMIT_DIST:
            continue
        Pr = np.vstack([p, wps])
        if collect:
            visited.append((Pr[:MAX_SEG + 1].copy(), v.copy(), a.copy()))
        traj = planner.plan(Pr, v, a, t)
    return traj.t_end, first_duration, visited


def _label(args):
    probs, seed = args
    lim = TrajLimits()
    out = dict(P=[], v0=[], a0=[], T=[], X=[], nseg=[], J=[])
    for P, v0, a0 in probs:
        if np.linalg.norm(P[1, :2] - P[0, :2]) < 0.05:
            continue
        Pc, vc, ac, _ = to_canonical(P, v0, a0)
        try:
            tr, scale = teacher(Pc, vc, ac, lim)
        except np.linalg.LinAlgError:
            continue
        if scale > 3.0 or not np.all(np.isfinite(tr.knots)):
            continue
        k = len(Pc)
        Pp = np.zeros((MAX_KNOTS, 3))
        Pp[:k] = Pc
        Tp = np.ones(MAX_SEG)
        Tp[:k - 1] = tr.T
        Xp = np.zeros((MAX_KNOTS, 4, 3))
        Xp[:k] = tr.knots
        for key, val in zip(out, (Pp, vc, ac, Tp, Xp, k - 1, tr.snap_cost())):
            out[key].append(val)
    return {k: np.array(v) for k, v in out.items()}


def _rollout_job(args):
    weights, seeds, collect = args
    import torch
    torch.set_num_threads(1)
    from ..planning.planners import LearnedPlanner, MinSnapPlanner
    planner = MinSnapPlanner() if weights == 'teacher' else LearnedPlanner(weights, 'full')
    res = []
    for s in seeds:
        c = sample_course(np.random.default_rng(s), with_obstacles=False)
        P = np.vstack([c.start, c.waypoints()])
        t_fin, t0, vis = receding_rollout(planner, P, collect=collect)
        res.append((t_fin, t0, vis))
    return res


def rollouts(weights, seeds, workers=6, collect=False):
    chunks = [(weights, list(seeds[i::workers]), collect) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_rollout_job, chunks)
    return [r for p in parts for r in p]


def label(problems, workers=6):
    chunks = [(problems[i::workers], i) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_label, chunks)
    return {k: np.concatenate([p[k] for p in parts if len(p['P'])]) for k in parts[0]}
