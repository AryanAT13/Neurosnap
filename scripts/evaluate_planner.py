"""Offline planner evaluation on held-out (re-)planning problems.

For every problem each planner produces a limit-respecting trajectory; we report
snap cost and total duration relative to the optimal-time teacher, limit violations
and per-call compute time.

    python scripts/evaluate_planner.py --n 1000
Outputs results/planner_eval.json and docs/figures/planner_eval.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import FIG_DIR, MODELS_DIR, RESULTS_DIR, TrajLimits  # noqa: E402
from neurosnap.planning.planners import LearnedPlanner, MinSnapPlanner  # noqa: E402
from neurosnap.config import DATA_DIR, PLANNER_TAG  # noqa: E402


def planners(lim):
    out = [('Min-snap QP, heuristic T', MinSnapPlanner(lim, optimal_times=False)),
           ('Min-snap QP, optimised T (teacher)', MinSnapPlanner(lim, optimal_times=True))]
    for tag, label in [('dagger', 'Transformer, no snap loss, DAgger'), ('transformer', 'Transformer, snap loss 0.5'), ('transformer_snap01', 'Transformer, snap loss 0.1'),
                       ('transformer_nosnap', 'Transformer, no snap loss'), ('mlp', 'MLP, snap loss 0.5')]:
        w = os.path.join(MODELS_DIR, f'planner_{tag}.pt')
        if os.path.exists(w):
            out.append((f'{label} (full neural)', LearnedPlanner(w, 'full', lim)))
            if tag == PLANNER_TAG:
                out.append((f'{label} + QP derivatives (hybrid)', LearnedPlanner(w, 'hybrid', lim)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    a = ap.parse_args()
    lim = TrajLimits()
    d = np.load(os.path.join(DATA_DIR, 'planner_test.npz'))
    n = min(a.n, len(d['P']))
    probs = []
    for i in range(n):
        k = int(d['nseg'][i]) + 1
        probs.append((d['P'][i, :k], d['v0'][i], d['a0'][i]))
    results = {}
    ref_J, ref_T = None, None
    for name, pl in planners(lim):
        J, T, viol, ms, start_err = [], [], [], [], []
        for P, v0, a0 in probs:
            t = time.perf_counter()
            tr = pl.plan(P, v0, a0)
            ms.append((time.perf_counter() - t) * 1e3)
            J.append(tr.snap_cost())
            T.append(tr.duration)
            vm, am = tr.max_vel_acc()
            viol.append(vm > lim.v_max * 1.02 or am > lim.a_max * 1.02)
            p, v, acc, _ = tr.state(0.0)
            start_err.append(np.linalg.norm(v - v0) + np.linalg.norm(acc - a0))
        J, T = np.array(J), np.array(T)
        if 'teacher' in name:
            ref_J, ref_T = J, T
        results[name] = dict(J=J, T=T, viol=float(np.mean(viol)), ms=np.array(ms),
                             start_err=float(np.max(start_err)))
        print(f'{name:42s} {np.mean(ms):7.2f} ms/call')
    summary = {}
    for name, r in results.items():
        jr = r['J'] / ref_J
        tr = r['T'] / ref_T
        summary[name] = dict(
            snap_ratio_median=float(np.median(jr)), snap_ratio_p90=float(np.percentile(jr, 90)),
            duration_ratio_median=float(np.median(tr)), duration_ratio_p90=float(np.percentile(tr, 90)),
            limit_violation_rate=r['viol'], ms_mean=float(r['ms'].mean()),
            ms_p95=float(np.percentile(r['ms'], 95)), max_start_state_err=r['start_err'])
    print(f'\n{"planner":42s} {"snap/opt (med)":>14s} {"snap p90":>9s} {"T/opt med":>9s} {"viol":>6s} {"ms":>7s} {"p95 ms":>7s}')
    for name, s in summary.items():
        print(f'{name:42s} {s["snap_ratio_median"]:14.2f} {s["snap_ratio_p90"]:9.2f} '
              f'{s["duration_ratio_median"]:9.3f} {100 * s["limit_violation_rate"]:5.1f}% '
              f'{s["ms_mean"]:7.2f} {s["ms_p95"]:7.2f}')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'planner_eval.json'), 'w') as f:
        json.dump(dict(n_problems=n, summary=summary), f, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(summary)
    def label_of(nm):
        base = ('QP heuristic T' if 'heuristic' in nm else 'QP optimal T\n(teacher)' if 'teacher' in nm
                else 'MLP' if nm.startswith('MLP') else 'TF DAgger' if 'DAgger' in nm
                else 'TF ' + nm.split(', ')[1].split(' (')[0].split(' +')[0])
        return base + ('\n+ QP (hybrid)' if 'hybrid' in nm else '')

    short = [label_of(nm) for nm in names]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    ax[0].boxplot([np.log10(results[nm]['J'] / ref_J) for nm in names], showfliers=False)
    ax[0].set_ylabel('log10(snap cost / teacher)')
    ax[0].set_title('Trajectory smoothness (lower is better)')
    ax[1].boxplot([results[nm]['T'] / ref_T for nm in names], showfliers=False)
    ax[1].set_ylabel('duration / teacher')
    ax[1].set_title('Trajectory duration (lower is faster)')
    ax[2].bar(range(1, len(names) + 1), [summary[nm]['ms_mean'] for nm in names], color='#4a7fb5')
    ax[2].set_yscale('log')
    ax[2].set_ylabel('ms per planning call (log)')
    ax[2].set_title('Compute time (single CPU thread)')
    for x in ax:
        x.set_xticks(range(1, len(names) + 1))
        x.set_xticklabels(short, fontsize=7.5, rotation=40, ha='right')
        x.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'planner_eval.png'), dpi=130)


if __name__ == '__main__':
    main()
