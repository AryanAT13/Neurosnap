"""Train GateNet on an infinite stream of domain-randomised, self-labelled renders.

    python scripts/train_perception.py --epochs 30 --steps 250
Outputs models/gatenet.pt and models/gatenet_log.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import MODELS_DIR  # noqa: E402
from neurosnap.perception.data import FixedGateData, OnlineGateData  # noqa: E402
from neurosnap.perception.net import GateNet, detection_loss  # noqa: E402


def to_dev(t, dev):
    return {k: v.to(dev) for k, v in t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--steps', type=int, default=250, help='iterations per epoch')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    net = GateNet().to(dev)
    print(f'GateNet {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params on {dev}')
    print('rendering validation set ...', flush=True)
    val = DataLoader(FixedGateData(600, seed=777), batch_size=50)
    train = DataLoader(OnlineGateData(seed=1), batch_size=a.bs, num_workers=a.workers,
                       persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    it = iter(train)
    log, best = [], np.inf
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        tr = []
        for _ in range(a.steps):
            img, tgt = next(it)
            loss, m = detection_loss(net(img.to(dev)), to_dev(tgt, dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tr.append(m)
        net.eval()
        vl = []
        with torch.no_grad():
            for img, tgt in val:
                vl.append(detection_loss(net(img.to(dev)), to_dev(tgt, dev))[1])
        trm = {k: float(np.mean([x[k] for x in tr])) for k in tr[0]}
        vm = {k: float(np.mean([x[k] for x in vl])) for k in vl[0]}
        score = vm['hm'] + vm['off'] + 4 * vm['cor']
        log.append(dict(epoch=ep + 1, train=trm, val=vm, time=time.time() - t0))
        if score < best:
            best = score
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, epoch=ep + 1, val=vm),
                       os.path.join(MODELS_DIR, 'gatenet.pt'))
        print(f'ep {ep + 1:3d}  train hm {trm["hm"]:.3f} cor {trm["cor"]:.4f} | val hm {vm["hm"]:.3f} '
              f'off {vm["off"]:.3f} cor {vm["cor"]:.4f}  ({time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(MODELS_DIR, 'gatenet_log.json'), 'w') as f:
        json.dump(dict(images_seen=a.epochs * a.steps * a.bs, log=log), f, indent=1)


if __name__ == '__main__':
    main()
