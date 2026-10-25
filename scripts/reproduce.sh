#!/bin/sh
# Full pipeline: data -> training -> evaluation -> benchmark -> figures -> demo.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}
mkdir -p logs
$PY scripts/generate_planner_data.py --train 80000 --test 4000
$PY scripts/train_planner.py --kind transformer
$PY scripts/train_planner.py --kind mlp
$PY scripts/train_planner.py --kind transformer --no-snap
$PY scripts/train_planner.py --kind transformer --w-snap 0.1 --tag transformer_snap01
$PY scripts/dagger_planner.py --iters 3            # deployed planner: models/planner_dagger.pt
$PY scripts/train_perception.py --epochs 25 --steps 250
$PY scripts/train_residual.py --flights 40
$PY scripts/evaluate_planner.py --n 1000
$PY scripts/evaluate_perception.py --n 1000
$PY scripts/benchmark.py --seeds 40 --workers 6
$PY scripts/make_figures.py
$PY scripts/demo.py --seed 3 --save docs/media/demo.mp4 --gif docs/media/demo.gif
$PY -m pytest -q tests
