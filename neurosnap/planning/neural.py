"""Neural trajectory generator: waypoints + current state -> segment durations + knot derivatives.

The network replaces both the time-allocation optimisation and the QP of the
minimum-snap planner. Its outputs are decoded by a fixed, differentiable layer
(the 7th-order Hermite parametrisation of `traj/poly.py`), so every predicted
trajectory passes exactly through the waypoints, starts from the current state
and is C3-continuous by construction; the network only has to get the shape right.

Two encoders are provided: a Transformer over the waypoint sequence (proposed) and
a flat MLP baseline (ablation).
"""
import numpy as np
import torch
import torch.nn as nn

from ..traj.dataset import MAX_KNOTS, MAX_SEG, to_canonical
from ..traj.minsnap import solve_fixed_times
from ..traj.poly import DPOW, H_MAT, T_EXP, PiecewisePoly, enforce_limits

POS_S, V_S, A_S = 3.0, 3.0, 6.0
N_FEAT = 16
N_OUT = 10              # 3 x (v, a, j) + log T of the incoming segment


def make_features(P, v0, a0, mask):
    """P (B, K, 3) canonical waypoints incl. start, mask (B, K) -> (B, K, 16) tokens."""
    B, K, _ = P.shape
    f = torch.zeros(B, K, N_FEAT, dtype=P.dtype)
    f[..., 0:3] = P / POS_S
    d = torch.zeros_like(P)
    d[:, 1:] = P[:, 1:] - P[:, :-1]
    f[..., 3:6] = d / POS_S
    f[..., 6] = d.norm(dim=-1) / POS_S
    f[:, 0, 7] = 1.0
    nseg = mask.sum(1).long() - 1
    f[torch.arange(B), nseg, 8] = 1.0
    f[:, 0, 9:12] = v0 / V_S
    f[:, 0, 12:15] = a0 / A_S
    f[..., 15] = torch.arange(K, dtype=P.dtype)[None] / MAX_SEG
    return f * mask[..., None]


class TransformerPlanner(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ff=256):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEAT, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, MAX_KNOTS, d))
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True,
                                         activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_OUT))

    def forward(self, feat, mask):
        h = self.inp(feat) + self.pos[:, :feat.shape[1]]
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.out(h)


class MLPPlanner(nn.Module):
    def __init__(self, hidden=512, layers=3):
        super().__init__()
        dims = [MAX_KNOTS * (N_FEAT + 1)] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*mods, nn.Linear(hidden, MAX_KNOTS * N_OUT))

    def forward(self, feat, mask):
        x = torch.cat([feat, mask[..., None].to(feat.dtype)], -1).flatten(1)
        return self.net(x).view(-1, MAX_KNOTS, N_OUT)


def build_model(kind, **kw):
    return TransformerPlanner(**kw) if kind == 'transformer' else MLPPlanner(**kw)


def knot_tau(P, mask, v_ref=1.8):
    """Geometric time scale of every knot: mean (distance / v_ref) of its adjacent segments.

    It depends only on the inputs, so the derivative outputs are decoupled from the
    predicted durations while still adapting to short and long segments.
    """
    seg = mask[:, 1:].to(P.dtype)
    Th = ((P[:, 1:] - P[:, :-1]).norm(dim=-1) / v_ref).clamp_min(0.25) * seg
    zero = torch.zeros_like(Th[:, :1])
    cnt = torch.cat([zero, seg], 1) + torch.cat([seg, zero], 1)
    tau = (torch.cat([zero, Th], 1) + torch.cat([Th, zero], 1)) / cnt.clamp_min(1.0)
    return tau + (cnt == 0).to(P.dtype)


DPOW_K = torch.tensor([1.0, 2.0, 3.0])
DER_STD = torch.tensor([2.5, 2.7, 5.4])          # std of x_d * tau^d on the training set (m)


def normalise_derivs(X, tau):
    """Knot derivatives (B, K, 3, 3) -> standardised, duration-normalised units."""
    scale = tau[..., None, None] ** DPOW_K.to(X)[None, None, :, None] / DER_STD.to(X)[None, None, :, None]
    return X * scale


def decode(out, P, v0, a0, mask):
    """Network output -> (T (B, S), knots (B, K, 4, 3)) with the hard constraints applied.

    Velocity, acceleration and jerk are predicted as x_d * tau^d (standardised), which
    puts all three orders on the same scale whatever the segment lengths are.
    """
    B, K, _ = P.shape
    logT = out[:, 1:, 9]
    T = torch.exp(logT.clamp(-3.0, 3.5))
    tau = knot_tau(P, mask)
    scale = tau[..., None, None] ** DPOW_K.to(out)[None, None, :, None] / DER_STD.to(out)[None, None, :, None]
    der = out[..., :9].view(B, K, 3, 3) / scale
    X = torch.zeros(B, K, 4, 3, dtype=out.dtype)
    X[:, :, 0] = P
    X[:, :, 1:] = der
    X[:, 0, 1] = v0
    X[:, 0, 2] = a0
    nseg = mask.sum(1).long() - 1
    end = torch.zeros(B, K, dtype=torch.bool)
    end[torch.arange(B), nseg] = True
    X[:, :, 1:] = X[:, :, 1:] * (~end & mask)[..., None, None]        # stop at the finish, zero padding
    return T, X


_H = torch.tensor(H_MAT, dtype=torch.float32)
_E = torch.tensor(T_EXP, dtype=torch.float32)


def snap_cost(T, X, seg_mask):
    """Integrated squared snap per sample. T (B, S), X (B, S+1, 4, 3), seg_mask (B, S)."""
    x = torch.cat([X[:, :-1], X[:, 1:]], dim=2)                       # (B, S, 8, 3)
    G = _H.to(T) * torch.pow(T[..., None, None], _E.to(T))            # (B, S, 8, 8)
    J = torch.einsum('bsic,bsij,bsjc->bs', x, G, x)
    return (J * seg_mask).sum(1)


class NeuralPlanner:
    """Run-time wrapper: world-frame waypoints in, PiecewisePoly out.

    mode='full'   network predicts durations and derivatives (no solver at run time)
    mode='hybrid' network predicts durations, derivatives from the closed-form QP
    """

    def __init__(self, path, mode='full', lim=None):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.model = build_model(ck['kind'], **ck.get('kwargs', {}))
        self.model.load_state_dict(ck['state'])
        self.model.eval()
        self.mode = mode
        self.lim = lim
        self.kind = ck['kind']

    @torch.no_grad()
    def predict(self, P, v0, a0):
        Pc, vc, ac, (o, R) = to_canonical(P, v0, a0)
        k = len(Pc)
        Pp = torch.zeros(1, MAX_KNOTS, 3)
        Pp[0, :k] = torch.tensor(Pc, dtype=torch.float32)
        mask = torch.zeros(1, MAX_KNOTS, dtype=torch.bool)
        mask[0, :k] = True
        v = torch.tensor(vc, dtype=torch.float32)[None]
        a = torch.tensor(ac, dtype=torch.float32)[None]
        out = self.model(make_features(Pp, v, a, mask), mask)
        T, X = decode(out, Pp, v, a, mask)
        T = T[0, :k - 1].double().numpy()
        X = X[0, :k].double().numpy()
        X = X @ R.T                                   # rotate vectors back (positions handled below)
        X[:, 0] = P
        return T, X

    def plan(self, P, v0, a0, t0=0.0):
        P = np.asarray(P, float)
        if len(P) - 1 > MAX_SEG:
            P = P[:MAX_SEG + 1]
        T, X = self.predict(P, v0, a0)
        if self.mode == 'hybrid':
            def resolve(Tn):
                return PiecewisePoly(Tn, solve_fixed_times(P, Tn, v0, a0)[0], t0)
            traj, _ = enforce_limits(resolve(T), self.lim, resolve)
        else:
            X[0, 1], X[0, 2] = v0, a0
            traj, _ = enforce_limits(PiecewisePoly(T, X, t0), self.lim)
        return traj
