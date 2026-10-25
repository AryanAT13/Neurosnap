"""Classical minimum-snap trajectory generation (Mellinger & Kumar, ICRA 2011).

This is the "complex mathematical" baseline the project replaces at run time. It is
used (1) offline as the teacher that labels the planner dataset and (2) online as
the comparison planners in the benchmark.

Given waypoints P_0..P_n and segment durations T_1..T_n, the free knot derivatives
(initial jerk and v, a, j at every interior waypoint) minimise the integrated squared
snap. With the knot-derivative parametrisation of `poly.py` this is an unconstrained
quadratic program with a closed-form solution (one linear solve, shared by x, y, z).

The segment durations are then optimised on log T with L-BFGS for
    F(T) = J*(T) + rho * sum(T),
whose gradient follows from the envelope theorem: dJ*/dT_s = x_s^T dG_s/dT x_s.
"""
import numpy as np
from scipy.optimize import minimize

from .poly import ORDER, PiecewisePoly, enforce_limits, seg_gram, seg_gram_dT


def _masks(n):
    """Boolean free-variable mask over the flattened (n+1)*4 knot vector."""
    free = np.ones((n + 1, ORDER), dtype=bool)
    free[:, 0] = False           # positions are waypoints
    free[0, 1:3] = False         # initial velocity / acceleration are the current state
    free[n, 1:] = False          # stop at the finish point (v = a = j = 0)
    return free.reshape(-1)


def solve_fixed_times(P, T, v0=None, a0=None):
    """Minimum-snap knot derivatives for fixed durations.

    P: (n+1, 3) waypoints, T: (n,) durations. Returns knots (n+1, 4, 3) and the snap cost.
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(T)
    N = (n + 1) * ORDER
    Gf = np.zeros((N, N))
    for s in range(n):
        sl = slice(s * ORDER, (s + 2) * ORDER)
        Gf[sl, sl] += seg_gram(T[s])
    X = np.zeros((n + 1, ORDER, 3))
    X[:, 0] = P
    if v0 is not None:
        X[0, 1] = v0
    if a0 is not None:
        X[0, 2] = a0
    free = _masks(n)
    Xf = X.reshape(N, 3)
    Gff = Gf[np.ix_(free, free)]
    Gfx = Gf[np.ix_(free, ~free)]
    z = np.linalg.solve(Gff, -Gfx @ Xf[~free])
    Xf = Xf.copy()
    Xf[free] = z
    J = float(np.einsum('ic,ij,jc->', Xf, Gf, Xf))
    return Xf.reshape(n + 1, ORDER, 3), J


def cost_and_grad(P, T, v0, a0):
    knots, J = solve_fixed_times(P, T, v0, a0)
    g = np.zeros(len(T))
    for s in range(len(T)):
        x = np.concatenate([knots[s], knots[s + 1]], axis=0)
        g[s] = np.einsum('ic,ij,jc->', x, seg_gram_dT(T[s]), x)
    return J, g, knots


def heuristic_times(P, lim, v0=None):
    """Distance / cruise speed, with extra time on the accelerating and stopping segments."""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    T = np.maximum(d / lim.v_cruise, 0.25)
    if v0 is None or np.linalg.norm(v0) < 0.5:
        T[0] *= 1.4
    T[-1] *= 1.4
    return T


def optimize_times(P, lim, v0=None, a0=None, T_init=None, max_iter=200):
    """Mellinger-style time allocation: minimise J*(T) + rho * sum(T) over log T."""
    P = np.asarray(P, dtype=float)
    v0 = np.zeros(3) if v0 is None else np.asarray(v0, float)
    a0 = np.zeros(3) if a0 is None else np.asarray(a0, float)
    T0 = heuristic_times(P, lim, v0) if T_init is None else np.asarray(T_init, float)

    def f(u):
        T = np.exp(u)
        J, g, _ = cost_and_grad(P, T, v0, a0)
        F = J + lim.rho * T.sum()
        return F, (g + lim.rho) * T

    res = minimize(f, np.log(T0), jac=True, method='L-BFGS-B',
                   bounds=[(np.log(0.05), np.log(30.0))] * len(T0),
                   options=dict(maxiter=max_iter, gtol=1e-6))
    return np.exp(res.x), res


def plan_minsnap(P, lim, v0=None, a0=None, optimal_times=True, t0=0.0):
    """Full classical planner: (optionally optimised) times, closed-form QP, limit scaling."""
    v0 = np.zeros(3) if v0 is None else np.asarray(v0, float)
    a0 = np.zeros(3) if a0 is None else np.asarray(a0, float)
    if optimal_times:
        T, _ = optimize_times(P, lim, v0, a0)
    else:
        T = heuristic_times(P, lim, v0)
    def resolve(Tn):
        return PiecewisePoly(Tn, solve_fixed_times(P, Tn, v0, a0)[0], t0)

    traj, _ = enforce_limits(resolve(T), lim, resolve)
    return traj
