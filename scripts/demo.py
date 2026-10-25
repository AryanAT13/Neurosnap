"""Fly one gate course with the full NeuroSnap stack and show / record the dashboard.

    python scripts/demo.py --seed 3 --live                 # live OpenCV window
    python scripts/demo.py --seed 3 --save docs/media/demo.mp4 --gif docs/media/demo.gif
    python scripts/demo.py --planner qp_opt                # compare with the classical planner
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import MODELS_DIR, QuadParams  # noqa: E402
from neurosnap.mission import Mission  # noqa: E402
from neurosnap.perception.detector import GatePerception  # noqa: E402
from neurosnap.sim.residual import ResidualModel  # noqa: E402
from neurosnap.viz.dashboard import Dashboard  # noqa: E402
from neurosnap.world.course import sample_course  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import make_planner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--planner', default='nn_full', choices=['nn_full', 'nn_hybrid', 'qp_opt', 'qp_heur', 'line'])
    ap.add_argument('--perception', default='cnn', choices=['cnn', 'blind', 'oracle'])
    ap.add_argument('--no-residual', action='store_true')
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--save', default=None, help='output .mp4')
    ap.add_argument('--gif', default=None, help='output .gif (downscaled)')
    a = ap.parse_args()
    course = sample_course(np.random.default_rng(20_000 + a.seed))
    perc = GatePerception(os.path.join(MODELS_DIR, 'gatenet.pt'))
    res = None if a.no_residual else ResidualModel(os.path.join(MODELS_DIR, 'residual.npz'), QuadParams().hover_rpm)
    title = f'NeuroSnap  |  planner {a.planner}  |  perception {a.perception}' + ('' if res else '  |  no residual')
    dash = Dashboard(course, title)
    frames = []
    writer = None

    def on_frame(mission, t, img, dets, traj, nxt):
        nonlocal writer
        canvas = dash.draw(mission, t, img, dets, traj, nxt)
        if a.save:
            if writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*'mp4v'), 15, canvas.shape[1::-1])
            writer.write(canvas)
        if a.gif:
            frames.append(cv2.resize(canvas, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
        if a.live:
            cv2.imshow('NeuroSnap', canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt

    m = Mission(course, make_planner(a.planner), a.perception, perception=perc, residual=res,
                seed=a.seed, on_frame=on_frame)
    r = m.run()
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    if writer is not None:
        writer.release()
        print('saved', a.save)
    if a.gif and frames:
        from PIL import Image
        imgs = [Image.fromarray(f[..., ::-1]) for f in frames[::2]]
        os.makedirs(os.path.dirname(os.path.abspath(a.gif)), exist_ok=True)
        imgs[0].save(a.gif, save_all=True, append_images=imgs[1:], duration=133, loop=0, optimize=True)
        print('saved', a.gif)
    if a.live:
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
