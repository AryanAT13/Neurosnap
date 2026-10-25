"""Piecewise 7th-order polynomial trajectories parametrised by knot derivatives.

Each segment is defined by position, velocity, acceleration and jerk at its two
end knots (8 numbers per axis), which fixes a unique 7th-order polynomial. On the
normalised interval tau = t / T in [0, 1] the boundary values become
y_d = x_d * T^d, the coefficients are a = A @ y with a constant matrix A, and the
integrated squared snap is

    J = int_0^T (d^4 p / dt^4)^2 dt = T^-7 * y^T H y,     H = A^T Q1 A.

So J is a quadratic form G(T) in the knot derivatives with G_ij = H_ij T^(d_i+d_j-7).
Both the minimum-snap teacher and the neural planner reuse this one parametrisation,
which guarantees that every trajectory passes through its waypoints and is C3-continuous.
"""
import math

import numpy as np

ORDER = 4                       # derivatives stored per knot: p, v, a, j
NCOEF = 8


def _constants():
    M = np.zeros((NCOEF, NCOEF))
    for d in range(ORDER):
        for k in range(d, NCOEF):
            c = math.factorial(k) / math.factorial(k - d)
            if k == d:
                M[d, k] = c                 # tau = 0
            M[ORDER + d, k] = c             # tau = 1
    A = np.linalg.inv(M)
    Q1 = np.zeros((NCOEF, NCOEF))
    for k in range(4, NCOEF):
        for l in range(4, NCOEF):
            ck = math.factorial(k) / math.factorial(k - 4)
            cl = math.factorial(l) / math.factorial(l - 4)
            Q1[k, l] = ck * cl / (k + l - 7)
    H = A.T @ Q1 @ A
    dpow = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    E = dpow[:, None] + dpow[None, :] - 7       # exponent of T in G(T)
    return A, H, dpow, E


A_MAT, H_MAT, DPOW, T_EXP = _constants()


def seg_gram(T):
    """8x8 snap Gram matrix G(T) of one segment (x^T G x = integrated squared snap)."""
    return H_MAT * np.power(T, T_EXP)


def seg_gram_dT(T):
    return H_MAT * T_EXP * np.power(T, T_EXP - 1)


class PiecewisePoly:
    """Trajectory through n+1 knots. knots: (n+1, 4, 3) array of [p, v, a, j] x xyz."""

    def __init__(self, T, knots, t0=0.0):
        self.T = np.asarray(T, dtype=float)
        self.knots = np.asarray(knots, dtype=float)
        self.t0 = t0
        self.tk = np.concatenate([[0.0], np.cumsum(self.T)])
        n = len(self.T)
        # y = [x0 * T^d, x1 * T^d] per segment -> normalised coefficients a (n, 8, 3)
        y = np.concatenate([self.knots[:-1], self.knots[1:]], axis=1)         # (n, 8, 3)
        y = y * np.power(self.T[:, None], DPOW)[:, :, None]
        self.coef = np.einsum('kj,njc->nkc', A_MAT, y)
        self.n = n
        self.stretch = 1.0          # time stretch applied by the safety layer

    @property
    def duration(self):
        return float(self.tk[-1])

    @property
    def t_end(self):
        return self.t0 + self.duration

    def _locate(self, t):
        tl = np.clip(np.atleast_1d(np.asarray(t, dtype=float)) - self.t0, 0.0, self.duration)
        i = np.clip(np.searchsorted(self.tk, tl, side='right') - 1, 0, self.n - 1)
        tau = (tl - self.tk[i]) / self.T[i]
        return i, tau

    def eval(self, t, order=0):
        """Derivative `order` (0..4) at times t; returns (len(t), 3)."""
        i, tau = self._locate(t)
        out = np.zeros((len(tau), 3))
        for k in range(order, NCOEF):
            c = math.factorial(k) / math.factorial(k - order)
            out += c * (tau ** (k - order))[:, None] * self.coef[i, k, :]
        return out / (self.T[i] ** order)[:, None]

    def state(self, t):
        """(p, v, a, j) at a single time t (clamped to the trajectory)."""
        return tuple(self.eval(t, d)[0] for d in range(4))

    def sample(self, dt=0.02):
        ts = self.t0 + np.arange(0.0, self.duration + 1e-9, dt)
        return ts, self.eval(ts, 0), self.eval(ts, 1), self.eval(ts, 2)

    def snap_cost(self):
        J = 0.0
        for s in range(self.n):
            x = np.concatenate([self.knots[s], self.knots[s + 1]], axis=0)   # (8, 3)
            J += float(np.einsum('ic,ij,jc->', x, seg_gram(self.T[s]), x))
        return J

    def max_vel_acc(self, dt=0.02):
        ts = self.t0 + np.linspace(0.0, self.duration, int(np.clip(self.duration / dt, 30, 1500)))
        return (float(np.linalg.norm(self.eval(ts, 1), axis=1).max()),
                float(np.linalg.norm(self.eval(ts, 2), axis=1).max()))

    def time_scaled(self, s, keep_start=True):
        """Same path flown s times slower (v/s, a/s^2, j/s^3).

        With keep_start the initial velocity / acceleration are kept exactly, so a
        re-planned trajectory still starts from the vehicle's current state.
        """
        k = self.knots.copy()
        for d in range(1, ORDER):
            k[:, d, :] /= s ** d
        if keep_start:
            k[0, 1:3] = self.knots[0, 1:3]
        return PiecewisePoly(self.T * s, k, self.t0)

    def shifted(self, t0):
        return PiecewisePoly(self.T, self.knots, t0)


def enforce_limits(traj, lim, resolve=None, iters=4):
    """Stretch the durations until max |v| and |a| respect the limits (never speeds up).

    resolve(T) -> PiecewisePoly re-optimises the knot derivatives for the new durations
    (used by the QP planners); otherwise the derivatives are time-scaled directly.
    Returns the trajectory and the applied overall scale factor.
    """
    total = 1.0
    for _ in range(iters):
        vm, am = traj.max_vel_acc()
        s = max(1.0, vm / lim.v_max, np.sqrt(am / lim.a_max))
        if s <= 1.0 + 1e-3:
            break
        s *= 1.02
        total *= s
        traj = resolve(traj.T * s) if resolve is not None else traj.time_scaled(s)
    traj.stretch = total
    return traj, total
