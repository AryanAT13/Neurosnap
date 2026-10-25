"""Label random (re-)planning problems with the minimum-snap teacher.

    python scripts/generate_planner_data.py --train 80000 --test 4000
Outputs data/planner_train.npz and data/planner_test.npz
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import DATA_DIR  # noqa: E402
from neurosnap.traj.dataset import generate  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=80000)
    ap.add_argument('--test', type=int, default=4000)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    t = time.time()
    generate(a.train, os.path.join(DATA_DIR, 'planner_train.npz'), seed=1, workers=a.workers)
    generate(a.test, os.path.join(DATA_DIR, 'planner_test.npz'), seed=2, workers=a.workers)
    print(f'done in {time.time() - t:.0f}s')
