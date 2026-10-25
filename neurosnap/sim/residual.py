"""Learned residual dynamics: an MLP that predicts the aerodynamic acceleration the
cascaded PID does not model (rotor drag), so the controller can cancel it as feed-forward.

Inputs  (body frame): velocity (3), summed rotor speed / hover value (1)
Output  (body frame): residual acceleration (3) = measured acc - (thrust / m - g)
Inference is done in numpy (two small layers) so it is cheap at the 500 Hz control rate.
"""
import numpy as np

from ..config import G


def features(v_world, R, rpm, hover_rpm):
    vb = R.T @ v_world
    return np.concatenate([vb / 3.0, [np.sum(rpm) / (4 * hover_rpm) - 1.0]])


def residual_target(acc_world, R, rpm, kf, mass):
    thrust_acc = R @ np.array([0.0, 0.0, kf * np.sum(rpm ** 2) / mass]) - np.array([0, 0, G])
    return R.T @ (acc_world - thrust_acc)


class ResidualModel:
    """numpy forward pass of the trained torch MLP (weights exported by train_residual.py)."""

    def __init__(self, path, hover_rpm):
        w = np.load(path)
        self.layers = [(w[f'W{i}'], w[f'b{i}']) for i in range(int(w['n']))]
        self.scale = float(w['out_scale'])
        self.hover = hover_rpm

    def __call__(self, v, R, rpm):
        x = features(v, R, rpm, self.hover)
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return R @ (x * self.scale)
