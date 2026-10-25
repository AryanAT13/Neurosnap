"""DAgger for the neural planner: fix the compounding slow-down of receding-horizon re-planning.

Each iteration rolls out the current network in planner-only receding-horizon mode on fresh
courses, labels the visited states with the minimum-snap teacher, adds them to the training
set and fine-tunes. Reports the drift ratio (receding-horizon time / first-plan duration).

    python scripts/dagger_planner.py --iters 3
Outputs data/dagger_<i>.npz, models/planner_dagger.pt, results/dagger.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import DATA_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from neurosnap.traj.dagger import label, rollouts  # noqa: E402

EVAL_SEEDS = list(range(70000, 70100))


def drift(weights):
    r = rollouts(weights, EVAL_SEEDS)
    ratio = np.array([min(a / b, 10.0) for a, b, _ in r])
    return dict(median=float(np.median(ratio)), p90=float(np.percentile(ratio, 90)),
                blowups=int(np.sum(ratio >= 3.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='transformer_nosnap')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--courses', type=int, default=1500)
    ap.add_argument('--per-iter', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=15)
    a = ap.parse_args()
    py = sys.executable
    cur = os.path.join(MODELS_DIR, f'planner_{a.base}.pt')
    hist = [dict(iter=0, model=os.path.basename(cur), teacher=drift('teacher'), drift=drift(cur))]
    print(hist[0], flush=True)
    extra = []
    rng = np.random.default_rng(0)
    for it in range(1, a.iters + 1):
        seeds = list(range(80000 + it * 10000, 80000 + it * 10000 + a.courses))
        visited = [p for _, _, vis in rollouts(cur, seeds, collect=True) for p in vis]
        pick = rng.choice(len(visited), min(a.per_iter, len(visited)), replace=False)
        data = label([visited[i] for i in pick])
        path = os.path.join(DATA_DIR, f'dagger_{it}.npz')
        np.savez_compressed(path, **data)
        extra.append(path)
        print(f'iter {it}: {len(visited)} visited states, {len(data["P"])} labelled', flush=True)
        tag = f'dagger_it{it}'
        subprocess.run([py, os.path.join(os.path.dirname(__file__), 'train_planner.py'), '--kind', 'transformer',
                        '--no-snap', '--init', cur, '--extra', *extra, '--epochs', str(a.epochs),
                        '--lr', '3e-4', '--tag', tag], check=True)
        cur = os.path.join(MODELS_DIR, f'planner_{tag}.pt')
        hist.append(dict(iter=it, model=os.path.basename(cur), states=len(data['P']), drift=drift(cur)))
        print(hist[-1], flush=True)
    shutil.copy(cur, os.path.join(MODELS_DIR, 'planner_dagger.pt'))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'dagger.json'), 'w') as f:
        json.dump(hist, f, indent=1)


if __name__ == '__main__':
    main()
