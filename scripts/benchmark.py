"""Closed-loop Monte-Carlo benchmark on unseen random gate courses.

Every method flies the same courses, prior maps and wind seeds. All re-planned trajectories
pass the analytic plan verifier (mission.accept_plan) unless stated otherwise.

  planner comparison (all with GateNet perception, plain PID)
    line+cnn          straight-line waypoint follower, 2.2 m/s
    line_fast+cnn     ... at 3.0 m/s (the min-snap planners' peak speed)
    qp_heur+cnn       min-snap QP, heuristic time allocation
    qp_opt+cnn        min-snap QP, optimised time allocation (reference paper)
    nn_hybrid+cnn     neural durations + closed-form QP derivatives
    nn_full+cnn       neural durations + neural derivatives (no solver), DAgger-trained
    nn_full_bc+cnn    same network before DAgger (behaviour cloning only)
  full stack and perception ablation (neural planner + learned residual tracker)
    neurosnap         nn_full + GateNet + residual model          <- proposed
    neurosnap_noverify  ... without the analytic plan verifier
    neurosnap_blind   ... flying the noisy prior map only
    neurosnap_oracle  ... with true gate positions (upper bound)

    python scripts/benchmark.py --seeds 40 --workers 6
Outputs results/benchmark.csv, results/benchmark_summary.json, docs/figures/benchmark*.png
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

METHODS = {
    'line+cnn': dict(planner='line', perception='cnn', residual=False),
    'line_fast+cnn': dict(planner='line_fast', perception='cnn', residual=False),
    'qp_heur+cnn': dict(planner='qp_heur', perception='cnn', residual=False),
    'qp_opt+cnn': dict(planner='qp_opt', perception='cnn', residual=False),
    'nn_hybrid+cnn': dict(planner='nn_hybrid', perception='cnn', residual=False),
    'nn_full+cnn': dict(planner='nn_full', perception='cnn', residual=False),
    'nn_full_bc+cnn': dict(planner='nn_full_bc', perception='cnn', residual=False),
    'neurosnap': dict(planner='nn_full', perception='cnn', residual=True),
    'neurosnap_noverify': dict(planner='nn_full', perception='cnn', residual=True, verify=False),
    'neurosnap_blind': dict(planner='nn_full', perception='blind', residual=True),
    'neurosnap_oracle': dict(planner='nn_full', perception='oracle', residual=True),
}
_CACHE = {}


def make_planner(kind):
    from neurosnap.config import MODELS_DIR, PLANNER_TAG
    from neurosnap.planning.planners import LearnedPlanner, MinSnapPlanner, StraightLinePlanner
    w = os.path.join(MODELS_DIR, f'planner_{PLANNER_TAG}.pt')
    if kind == 'nn_full_bc':        # behaviour cloning only, before DAgger
        return LearnedPlanner(os.path.join(MODELS_DIR, 'planner_transformer_nosnap.pt'), 'full', name='neural_full_bc')
    if kind == 'line':
        return StraightLinePlanner(v_cruise=2.2)
    if kind == 'line_fast':
        return StraightLinePlanner(v_cruise=3.0, acc=5.0, name='straight_line_fast')
    if kind == 'qp_heur':
        return MinSnapPlanner(optimal_times=False)
    if kind == 'qp_opt':
        return MinSnapPlanner(optimal_times=True)
    return LearnedPlanner(w, 'hybrid' if kind == 'nn_hybrid' else 'full')


def run_one(job):
    import torch
    torch.set_num_threads(1)
    from neurosnap.config import MODELS_DIR, QuadParams
    from neurosnap.mission import Mission
    from neurosnap.world.course import sample_course
    name, seed, record = job
    spec = METHODS[name]
    if spec['perception'] == 'cnn' and 'perc' not in _CACHE:
        from neurosnap.perception.detector import GatePerception
        _CACHE['perc'] = GatePerception(os.path.join(MODELS_DIR, 'gatenet.pt'))
    if spec['residual'] and 'res' not in _CACHE:
        from neurosnap.sim.residual import ResidualModel
        _CACHE['res'] = ResidualModel(os.path.join(MODELS_DIR, 'residual.npz'), QuadParams().hover_rpm)
    course = sample_course(np.random.default_rng(10_000 + seed))
    m = Mission(course, make_planner(spec['planner']), spec['perception'],
                perception=_CACHE.get('perc') if spec['perception'] == 'cnn' else None,
                residual=_CACHE.get('res') if spec['residual'] else None, seed=seed,
                verify=spec.get('verify', True))
    t = time.time()
    r = m.run(record=record)
    r['wall_s'] = time.time() - t
    r['method'], r['seed'] = name, seed
    if record:
        lg = r.pop('log')
        r['traj'] = dict(p=lg['p'][::5].tolist(),
                         gates=[[*g.center.tolist(), g.yaw] for g in course.gates],
                         prior=[[*g.center.tolist(), g.yaw] for g in m.prior],
                         start=course.start.tolist(), finish=course.finish.tolist())
    return r


def summarise(rows):
    out = {}
    for name in METHODS:
        rs = [r for r in rows if r['method'] == name]
        if not rs:
            continue
        ok = [r for r in rs if r['success']]

        def mean(key, src):
            v = [r[key] for r in src if r[key] is not None]
            return float(np.mean(v)) if v else None

        out[name] = dict(
            missions=len(rs), success=len(ok) / len(rs),
            crash=float(np.mean([r['outcome'] == 'crash' for r in rs])),
            missed=float(np.mean([r['outcome'] == 'missed' for r in rs])),
            timeout=float(np.mean([r['outcome'] == 'timeout' for r in rs])),
            gates_frac=float(np.mean([r['gates_passed'] / r['n_gates'] for r in rs])),
            lap_time=mean('lap_time', ok), track_rmse=mean('track_rmse', rs),
            pass_offset=mean('pass_offset', rs), gate_est_err=mean('gate_est_err', rs),
            prior_err=mean('prior_err', rs), plan_ms_mean=mean('plan_ms_mean', rs),
            plan_ms_p95=float(np.percentile([r['plan_ms_p95'] for r in rs], 95)),
            perc_ms=mean('perc_ms', rs),
            reject_rate=float(np.sum([r['n_rejected'] for r in rs]) / max(np.sum([r['n_plans'] for r in rs]), 1)))
    return out


def plot(summary, rows, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(summary)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    ax[0].bar(x, [100 * summary[n]['success'] for n in names], color='#3c8d5a')
    ax[0].set_ylabel('success rate (%)')
    ax[0].set_ylim(0, 105)
    ax[0].set_title('Mission success')
    ax[1].bar(x, [100 * summary[n]['gates_frac'] for n in names], color='#4a7fb5')
    ax[1].set_ylabel('gates passed (%)')
    ax[1].set_ylim(0, 105)
    ax[1].set_title('Gates passed')
    ax[2].bar(x, [100 * (summary[n]['track_rmse'] or 0) for n in names], color='#c98a2b')
    ax[2].set_yscale('log')
    ax[2].set_ylabel('tracking RMSE (cm, log)')
    ax[2].set_title('Trajectory tracking error')
    ax[3].bar(x, [summary[n]['plan_ms_mean'] or 0 for n in names], color='#8e5aa8')
    ax[3].set_ylabel('ms per (re-)plan')
    ax[3].set_title('Planner compute time')
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        a.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'benchmark.png'), dpi=130)

    rec = [r for r in rows if 'traj' in r]
    if rec:
        fig, ax = plt.subplots(figsize=(11, 6))
        tr0 = rec[0]['traj']
        for gx, gy, gz, yaw in tr0['gates']:
            d = 0.55 * np.array([-np.sin(yaw), np.cos(yaw)])
            ax.plot([gx - d[0], gx + d[0]], [gy - d[1], gy + d[1]], 'k-', lw=4)
        for gx, gy, gz, yaw in tr0['prior']:
            ax.plot(gx, gy, 'x', color='gray', ms=8)
        cols = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, r in enumerate(rec):
            p = np.array(r['traj']['p'])
            ax.plot(p[:, 0], p[:, 1], '-', color=cols[i % 10], lw=1.4, label=f"{r['method']} ({r['outcome']})")
        ax.plot(*tr0['start'][:2], 'go', ms=8)
        ax.plot(*tr0['finish'][:2], 'r*', ms=12)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
        ax.set_title('Same course, all methods (black: true gates, x: prior map)')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'benchmark_trajectories.png'), dpi=130)


def main():
    from neurosnap.config import FIG_DIR, RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--methods', nargs='*', default=list(METHODS))
    ap.add_argument('--plot-only', action='store_true', help='re-draw figures from saved results')
    a = ap.parse_args()
    if a.plot_only:
        summary = json.load(open(os.path.join(RESULTS_DIR, 'benchmark_summary.json')))['summary']
        rows = [dict(method=t['method'], outcome=t['outcome'], traj=t)
                for t in json.load(open(os.path.join(RESULTS_DIR, 'benchmark_trajectories.json')))]
        plot(summary, rows, FIG_DIR)
        return
    jobs = [(m, s, s == 0) for s in range(a.seeds) for m in a.methods]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            rows.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f'{i + 1}/{len(jobs)} missions ({time.time() - t0:.0f}s)', flush=True)
    rows.sort(key=lambda r: (list(METHODS).index(r['method']), r['seed']))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    keys = ['method', 'seed', 'outcome', 'success', 'gates_passed', 'n_gates', 'lap_time', 'track_rmse',
            'pass_offset', 'gate_est_err', 'prior_err', 'plan_ms_mean', 'plan_ms_p95', 'n_plans', 'n_rejected', 'perc_ms',
            'flight_time', 'wall_s']
    with open(os.path.join(RESULTS_DIR, 'benchmark.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    with open(os.path.join(RESULTS_DIR, 'benchmark_summary.json'), 'w') as f:
        json.dump(dict(seeds=a.seeds, summary=summary), f, indent=1)
    with open(os.path.join(RESULTS_DIR, 'benchmark_trajectories.json'), 'w') as f:
        json.dump([r['traj'] | dict(method=r['method'], outcome=r['outcome']) for r in rows if 'traj' in r], f)
    plot(summary, rows, FIG_DIR)
    print(f'\n{"method":18s} {"success":>8s} {"crash":>6s} {"missed":>7s} {"gates":>6s} {"lap s":>6s} '
          f'{"rmse cm":>8s} {"est err":>8s} {"plan ms":>8s}')
    for n, s in summary.items():
        lap = f'{s["lap_time"]:6.1f}' if s['lap_time'] else '     -'
        est = f'{s["gate_est_err"]:8.3f}' if s['gate_est_err'] is not None else '       -'
        print(f'{n:18s} {100 * s["success"]:7.1f}% {100 * s["crash"]:5.1f}% {100 * s["missed"]:6.1f}% '
              f'{100 * s["gates_frac"]:5.1f}% {lap} {100 * s["track_rmse"]:8.2f} {est} {s["plan_ms_mean"]:8.2f}')


if __name__ == '__main__':
    main()
