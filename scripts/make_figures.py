"""Architecture diagram, training curves and an FPV sample strip for the README / report.

    python scripts/make_figures.py
Outputs docs/figures/architecture.png, training_curves.png, fpv_samples.png
"""
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import FIG_DIR, MODELS_DIR  # noqa: E402
from neurosnap.geometry import quat_from_euler, quat_to_rot  # noqa: E402
from neurosnap.vision.renderer import Renderer, default_style  # noqa: E402
from neurosnap.world.course import sample_course  # noqa: E402


def architecture():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, title, sub, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=col, ec='#444', lw=1))
        ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x + w / 2, y + h * 0.3, sub, ha='center', va='center', fontsize=8)

    def arr(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, color='#444', lw=1))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha='center', fontsize=7, style='italic')

    blue, orange, green, purple = '#dde8f5', '#fbe8c8', '#dcefdc', '#ece0f3'
    box(0.2, 4.3, 2.2, 1.2, 'Quadrotor sim', '6-DOF CF2X, motor lag,\nrotor drag, wind', blue)
    box(3.0, 4.3, 2.2, 1.2, 'FPV renderer', '192x144 RGB @ 15 Hz\nray-cast ground + polygons', blue)
    box(5.8, 4.3, 2.2, 1.2, 'GateNet (CNN)', 'centre heatmap +\n4 corner keypoints', orange)
    box(8.6, 4.3, 1.9, 1.2, 'PnP', 'IPPE-square,\nknown 0.8 m gate', orange)
    box(10.9, 4.3, 1.9, 1.2, 'Gate map KF', 'prior map +\nrange-weighted fixes', orange)
    box(10.9, 2.3, 1.9, 1.2, 'Neural planner', 'Transformer:\nwaypoints -> T, v/a/j', green)
    box(8.3, 2.3, 2.2, 1.2, 'Hermite decoder', '7th-order, C3, exact\nwaypoints + start state', green)
    box(5.8, 2.3, 2.1, 1.2, 'Safety layer', 'v / a limits,\ntime stretching', green)
    box(3.0, 2.3, 2.4, 1.2, 'Cascaded PID', 'pos -> att -> rate\n+ learned drag FF', green)
    box(0.2, 2.3, 2.2, 1.2, 'Mixer', 'thrust + torques\n-> motor RPM', blue)
    for x1, x2 in [(2.4, 3.0), (5.2, 5.8), (8.0, 8.6), (10.5, 10.9)]:
        arr(x1, 4.9, x2, 4.9)
    arr(11.85, 4.3, 11.85, 3.5, 'gate estimates')
    for x1, x2 in [(10.9, 10.5), (8.3, 7.9), (5.8, 5.4), (3.0, 2.4)]:
        arr(x1, 2.9, x2, 2.9)
    arr(1.3, 3.5, 1.3, 4.3)
    box(0.2, 0.3, 3.4, 1.2, 'Min-snap teacher (offline)', 'Mellinger & Kumar QP + time\nallocation, 80k problems', purple)
    box(4.2, 0.3, 3.2, 1.2, 'Imitation training', 'log-T + derivative loss\n+ snap-gap loss', purple)
    box(8.0, 0.3, 4.8, 1.2, 'Self-labelled renders + flight logs', 'domain-randomised frames -> GateNet\n(v, rpm) -> drag residual MLP', purple)
    arr(3.6, 0.9, 4.2, 0.9)
    arr(7.4, 1.2, 11.2, 2.3, 'weights')
    ax.set_title('NeuroSnap: vision-guided learned trajectory generation (online loop top, offline training bottom)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'architecture.png'), dpi=140)


def training_curves():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for tag, col in [('transformer', '#3c6fb0'), ('transformer_snap01', '#3c8d5a'), ('mlp', '#c98a2b'),
                     ('transformer_nosnap', '#7a7a7a')]:
        p = os.path.join(MODELS_DIR, f'planner_{tag}_log.json')
        if not os.path.exists(p):
            continue
        lg = json.load(open(p))['log']
        ep = [e['epoch'] for e in lg]
        ax[0].plot(ep, [100 * e['val']['T_rel'] for e in lg], color=col, label=tag)
        ax[1].plot(ep, [e['val']['gap'] for e in lg], color=col, label=tag)
    ax[0].set_ylabel('segment-duration error (%)')
    ax[1].set_ylabel('snap-cost gap  log(J / J*)')
    for a in ax[:2]:
        a.set_xlabel('epoch')
        a.grid(alpha=0.3)
        a.legend()
    ax[0].set_title('Planner: time allocation (val)')
    ax[1].set_title('Planner: derivative optimality (val)')
    p = os.path.join(MODELS_DIR, 'gatenet_log.json')
    if os.path.exists(p):
        lg = json.load(open(p))['log']
        ep = [e['epoch'] for e in lg]
        ax[2].plot(ep, [e['val']['hm'] for e in lg], label='heatmap focal')
        ax[2].plot(ep, [10 * e['val']['cor'] for e in lg], label='corner L1 x10')
        ax[2].set_xlabel('epoch')
        ax[2].grid(alpha=0.3)
        ax[2].legend()
        ax[2].set_title('GateNet (val)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'training_curves.png'), dpi=130)


def fpv_samples():
    rng = np.random.default_rng(20_003)
    course = sample_course(rng)
    r = Renderer()
    tiles = []
    for g in course.gates[:4]:
        p = g.center - 2.8 * g.normal + np.array([0.3, -0.2, 0.1])
        d = g.center - p
        R = quat_to_rot(quat_from_euler(0.05, 0.12, np.arctan2(d[1], d[0])))
        img, _ = r.render(course, p, R, default_style(), with_labels=False)
        tiles.append(cv2.resize(img, (384, 288)))
    cv2.imwrite(os.path.join(FIG_DIR, 'fpv_samples.png'), np.hstack(tiles))


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    architecture()
    training_curves()
    fpv_samples()
    print('figures written to', FIG_DIR)
