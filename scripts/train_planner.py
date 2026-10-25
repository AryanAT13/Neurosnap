"""Train the neural trajectory generator by imitation of the minimum-snap teacher.

    python scripts/train_planner.py --kind transformer            # proposed
    python scripts/train_planner.py --kind mlp                    # ablation: flat MLP
    python scripts/train_planner.py --kind transformer --no-snap  # ablation: no snap loss

Loss = MSE(log T) + SmoothL1(knot derivatives) + w_snap * log(J(T*, d_pred) / J(T*, d*)),
where the last term is the snap-cost gap of the predicted derivatives at the teacher's
durations (>= 0, zero only for the optimal derivatives).
Outputs models/planner_<tag>.pt and models/planner_<tag>_log.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import DATA_DIR, MODELS_DIR  # noqa: E402
from neurosnap.planning.neural import build_model, decode, knot_tau, make_features, normalise_derivs, snap_cost  # noqa: E402
from neurosnap.traj.dataset import MAX_KNOTS  # noqa: E402


def load(*paths):
    ds = [np.load(p) for p in paths]
    t = {k: torch.tensor(np.concatenate([d[k] for d in ds]), dtype=torch.float32)
         for k in ('P', 'v0', 'a0', 'T', 'X', 'J')}
    t['nseg'] = torch.tensor(np.concatenate([d['nseg'] for d in ds]), dtype=torch.long)
    t['mask'] = torch.arange(MAX_KNOTS)[None] <= t['nseg'][:, None]
    return t


def batch_loss(model, b, w_snap):
    P, v0, a0, mask, nseg = b['P'], b['v0'], b['a0'], b['mask'], b['nseg']
    B = P.shape[0]
    out = model(make_features(P, v0, a0, mask), mask)
    T_pred, X_pred = decode(out, P, v0, a0, mask)
    seg = mask[:, 1:]
    # durations
    l_T = ((torch.log(T_pred) - torch.log(b['T'])) ** 2 * seg).sum() / seg.sum()
    # free derivatives: initial jerk and (v, a, j) of interior knots
    free = torch.zeros(B, MAX_KNOTS, 3, dtype=torch.bool)
    free[:, 0, 2] = True
    interior = mask.clone()
    interior[:, 0] = False
    interior[torch.arange(B), nseg] = False
    free |= interior[..., None]
    # compare in standardised, duration-normalised units
    tau = knot_tau(P, mask)
    err = F.smooth_l1_loss(normalise_derivs(X_pred[:, :, 1:], tau), normalise_derivs(b['X'][:, :, 1:], tau),
                           reduction='none', beta=0.05)
    l_d = (err * free[..., None]).sum() / (free.sum() * 3)
    # snap-cost gap at teacher durations
    J_pred = snap_cost(b['T'], X_pred, seg)
    J_star = snap_cost(b['T'], b['X'], seg).clamp_min(1e-4)
    gap = torch.log(J_pred.clamp_min(1e-4) / J_star)
    l_snap = gap.mean()
    loss = 3.0 * l_T + l_d + w_snap * l_snap
    return loss, dict(l_T=l_T.item(), l_d=l_d.item(), gap=gap.mean().item(),
                      T_rel=((T_pred / b['T'] - 1).abs() * seg).sum().item() / seg.sum().item(),
                      sumT_rel=((T_pred * seg).sum(1) / (b['T'] * seg).sum(1) - 1).abs().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='transformer', choices=['transformer', 'mlp'])
    ap.add_argument('--no-snap', action='store_true')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--w-snap', type=float, default=0.5)
    ap.add_argument('--init', default=None, help='checkpoint to fine-tune from (DAgger)')
    ap.add_argument('--extra', nargs='*', default=[], help='additional training sets (DAgger data)')
    a = ap.parse_args()
    tag = a.tag or (a.kind + ('_nosnap' if a.no_snap else ''))
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tr = load(os.path.join(DATA_DIR, 'planner_train.npz'), *a.extra)
    te = load(os.path.join(DATA_DIR, 'planner_test.npz'))
    model = build_model(a.kind)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['state'])
    n_par = sum(p.numel() for p in model.parameters())
    print(f'{tag}: {n_par / 1e3:.0f}k parameters, {len(tr["P"])} train / {len(te["P"])} test problems')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    N = len(tr['P'])
    steps = a.epochs * (N // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    w_snap = 0.0 if a.no_snap else a.w_snap
    log, best = [], np.inf
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(N)
        agg = []
        for i in range(N // a.bs):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            b = {k: v[idx] for k, v in tr.items()}
            loss, m = batch_loss(model, b, w_snap)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg.append(m)
        model.eval()
        with torch.no_grad():
            vm = [batch_loss(model, {k: v[j:j + 1000] for k, v in te.items()}, w_snap)[1]
                  for j in range(0, len(te['P']), 1000)]
        trm = {k: float(np.mean([x[k] for x in agg])) for k in agg[0]}
        vam = {k: float(np.mean([x[k] for x in vm])) for k in vm[0]}
        log.append(dict(epoch=ep + 1, train=trm, val=vam, time=time.time() - t0))
        score = vam['l_T'] + vam['l_d'] + 0.2 * vam['gap']  # same model-selection rule for all variants
        if score < best or a.init:      # fine-tuning: keep the final weights
            best = score
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(dict(kind=a.kind, state=model.state_dict(), epoch=ep + 1, val=vam),
                       os.path.join(MODELS_DIR, f'planner_{tag}.pt'))
        print(f'ep {ep + 1:3d}  train l_T {trm["l_T"]:.4f} l_d {trm["l_d"]:.4f} gap {trm["gap"]:.3f} | '
              f'val l_T {vam["l_T"]:.4f} l_d {vam["l_d"]:.4f} gap {vam["gap"]:.3f} '
              f'|T err| {vam["T_rel"] * 100:.1f}% |sumT err| {vam["sumT_rel"] * 100:.1f}%  ({time.time() - t0:.0f}s)',
              flush=True)
    with open(os.path.join(MODELS_DIR, f'planner_{tag}_log.json'), 'w') as f:
        json.dump(dict(tag=tag, params=n_par, log=log), f, indent=1)


if __name__ == '__main__':
    main()
