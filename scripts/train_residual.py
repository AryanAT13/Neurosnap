"""Collect flight data and train the residual-dynamics MLP (learned drag feed-forward).

    python scripts/train_residual.py --flights 40
Flies minimum-snap reference trajectories over random courses with the plain cascaded
PID, logs (body velocity, rotor speed) -> residual acceleration, fits a small MLP and
exports it for numpy inference. Outputs models/residual.npz and models/residual_log.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import Config, MODELS_DIR  # noqa: E402
from neurosnap.sim.controller import CascadedPID  # noqa: E402
from neurosnap.sim.quadrotor import Quadrotor  # noqa: E402
from neurosnap.sim.residual import features, residual_target  # noqa: E402
from neurosnap.traj.minsnap import plan_minsnap  # noqa: E402
from neurosnap.world.course import sample_course  # noqa: E402


def fly(seed, cfg, speed_scale):
    rng = np.random.default_rng(seed)
    course = sample_course(rng, with_obstacles=False)
    lim = cfg.lim
    P = np.vstack([course.start, course.waypoints()])
    traj = plan_minsnap(P, lim)
    if speed_scale != 1.0:
        traj = traj.time_scaled(speed_scale, keep_start=False)
    q = Quadrotor(cfg.quad, wind=np.concatenate([rng.normal(0, 0.3, 2), [0]]))
    q.reset(course.start)
    ctrl = CascadedPID(q, cfg.ctrl)
    dt = 1 / cfg.sim_hz
    X, Y = [], []
    rpm = np.full(4, cfg.quad.hover_rpm)
    for k in range(int(traj.duration * cfg.sim_hz)):
        t = k * dt
        if k % 2 == 0:
            p, v, a, _ = traj.state(t)
            d = traj.state(min(t + 0.3, traj.t_end))[0] - q.p
            yaw = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d[:2]) > 0.2 else 0.0
            rpm = ctrl.compute(q.p, q.v, q.R, q.w, dict(p=p, v=v, a=a, yaw=yaw), 2 * dt)
        q.step(rpm, dt)
        if k % 2 == 0:
            X.append(features(q.v, q.R, q.rpm, cfg.quad.hover_rpm))
            Y.append(residual_target(q.acc, q.R, q.rpm, cfg.quad.kf, cfg.quad.mass))
    return np.array(X), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flights', type=int, default=40)
    ap.add_argument('--epochs', type=int, default=30)
    a = ap.parse_args()
    cfg = Config()
    Xs, Ys = [], []
    for i in range(a.flights):
        X, Y = fly(5000 + i, cfg, speed_scale=[1.0, 0.8, 1.25][i % 3])
        Xs.append(X)
        Ys.append(Y)
        if (i + 1) % 10 == 0:
            print(f'flight {i + 1}/{a.flights}: {sum(len(x) for x in Xs)} samples', flush=True)
    n_val = max(1, a.flights // 8)
    Xtr, Ytr = np.concatenate(Xs[n_val:]), np.concatenate(Ys[n_val:])
    Xva, Yva = np.concatenate(Xs[:n_val]), np.concatenate(Ys[:n_val])
    scale = float(np.abs(Ytr).max())
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 3))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr / scale, dtype=torch.float32)
    xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(Yva / scale, dtype=torch.float32)
    log = []
    for ep in range(a.epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 1024):
            idx = perm[i:i + 1024]
            loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            rmse = float(((net(xv) - yv) ** 2).mean().sqrt()) * scale
        log.append(dict(epoch=ep + 1, val_rmse=rmse))
    base = float(np.sqrt(np.mean(Yva ** 2)))
    print(f'residual acc RMS {base:.3f} m/s^2 -> model error {rmse:.3f} m/s^2 '
          f'({100 * (1 - rmse / base):.0f}% explained)')
    lin = [m for m in net if isinstance(m, nn.Linear)]
    w = {f'W{i}': m.weight.detach().numpy() for i, m in enumerate(lin)}
    w.update({f'b{i}': m.bias.detach().numpy() for i, m in enumerate(lin)})
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.savez(os.path.join(MODELS_DIR, 'residual.npz'), n=len(lin), out_scale=scale, **w)
    with open(os.path.join(MODELS_DIR, 'residual_log.json'), 'w') as f:
        json.dump(dict(samples=len(Xtr), residual_rms=base, model_rmse=rmse, log=log), f, indent=1)


if __name__ == '__main__':
    main()
