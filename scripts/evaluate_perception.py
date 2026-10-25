"""Evaluate GateNet + PnP on held-out renders.

Metrics: detection precision / recall (a detection matches a gate if its centre is
within 6 px and its corner error below 25% of the gate's image size), corner pixel
error, PnP centre position error vs. range, and inference latency.

    python scripts/evaluate_perception.py --n 1000
Outputs results/perception_eval.json, docs/figures/perception_eval.png and
docs/figures/perception_examples.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import FIG_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from neurosnap.perception.data import make_sample  # noqa: E402
from neurosnap.perception.detector import GatePerception  # noqa: E402
from neurosnap.perception.gatemap import measurement_sigma  # noqa: E402
from neurosnap.vision.renderer import Renderer  # noqa: E402


def match(dets, labels):
    used, pairs = set(), []
    for d in sorted(dets, key=lambda x: -x['score']):
        best, bd = None, 6.0
        for j, l in enumerate(labels):
            if j in used:
                continue
            e = np.linalg.norm(d['center_uv'] - l['center_uv'])
            if e < bd:
                best, bd = j, e
        if best is None:
            pairs.append((d, None))
            continue
        l = labels[best]
        size = np.ptp(l['corners_uv'], axis=0).max()
        cerr = np.linalg.norm(d['corners_uv'] - l['corners_uv'], axis=1).mean()
        if cerr < 0.25 * max(size, 8):
            used.add(best)
            pairs.append((d, l))
        else:
            pairs.append((d, None))
    return pairs, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    a = ap.parse_args()
    perc = GatePerception(os.path.join(MODELS_DIR, 'gatenet.pt'))
    rng = np.random.default_rng(424242)
    r = Renderer(seed=424242)
    tp = fp = n_gt = 0
    rows = []                # (range, corner px error, position error)
    ms = []
    examples = []
    for i in range(a.n):
        img, labels = make_sample(rng, r)
        # score the gates the planner relies on (within 8 m and at least ~10 px wide); detections
        # of farther / tinier gates are ignored rather than counted as false positives
        in_scope = [l['dist'] < 8.0 and np.ptp(l['corners_uv'], axis=0).max() > 10 for l in labels]
        dets = perc.detect(img)
        ms.append(perc.last_ms)
        pairs, used = match(dets, labels)
        n_gt += int(sum(in_scope))
        for d, l in pairs:
            if l is None:
                fp += 1
                continue
            if not in_scope[labels.index(l)]:
                continue
            tp += 1
            rows.append((l['dist'], float(np.linalg.norm(d['corners_uv'] - l['corners_uv'], axis=1).mean()),
                         float(np.linalg.norm(d['t_cam'] - l['center_cam']))))
        labels = [l for l, k in zip(labels, in_scope) if k]
        if len(examples) < 12 and labels:
            vis = img.copy()
            for d in dets:
                cv2.polylines(vis, [np.round(d['corners_uv']).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
                c = tuple(int(v) for v in d['center_uv'])
                cv2.putText(vis, f'{np.linalg.norm(d["t_cam"]):.1f}m', (c[0] + 3, c[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            examples.append(cv2.resize(vis, (384, 288), interpolation=cv2.INTER_LINEAR))
    rows = np.array(rows)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    bins = [0, 2, 4, 6, 8]
    by_range = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum():
            by_range.append(dict(range=f'{lo}-{hi} m', n=int(m.sum()),
                                 corner_px=float(np.median(rows[m, 1])),
                                 pos_err_median=float(np.median(rows[m, 2])),
                                 pos_err_p90=float(np.percentile(rows[m, 2], 90)),
                                 model_sigma=float(measurement_sigma((lo + hi) / 2))))
    summary = dict(frames=a.n, gates=n_gt, precision=prec, recall=rec,
                   corner_px_median=float(np.median(rows[:, 1])),
                   pos_err_median=float(np.median(rows[:, 2])),
                   latency_ms_mean=float(np.mean(ms)), latency_ms_p95=float(np.percentile(ms, 95)),
                   by_range=by_range)
    print(json.dumps(summary, indent=1))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'perception_eval.json'), 'w') as f:
        json.dump(summary, f, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(rows[:, 0], rows[:, 2], s=4, alpha=0.3, color='#4a7fb5', label='PnP fix')
    d = np.linspace(0.5, 8, 50)
    ax[0].plot(d, measurement_sigma(d), 'r-', label='KF measurement sigma')
    ax[0].set_ylim(0, 1.5)
    ax[0].set_xlabel('range to gate (m)')
    ax[0].set_ylabel('gate-centre position error (m)')
    ax[0].set_title(f'PnP error vs range (P={prec:.2f}, R={rec:.2f})')
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].hist(rows[:, 1], bins=50, range=(0, 6), color='#4a7fb5')
    ax[1].set_xlabel('mean corner error (px)')
    ax[1].set_title('Corner keypoint error')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'perception_eval.png'), dpi=130)
    grid = np.vstack([np.hstack(examples[i * 4:(i + 1) * 4]) for i in range(len(examples) // 4)])
    cv2.imwrite(os.path.join(FIG_DIR, 'perception_examples.png'), grid)


if __name__ == '__main__':
    main()
