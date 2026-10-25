"""Gate map: Kalman filter over gate centres, initialised from the noisy prior track map.

Each detection (PnP centre transformed into the world frame) is associated with the
nearest not-yet-passed gate by Mahalanobis distance and fused with a range-dependent
measurement covariance (PnP depth error grows roughly with distance squared).
"""
import numpy as np

from ..config import CourseParams


def measurement_sigma(dist):
    """1-sigma position error of a PnP gate fix at a given range (fitted from evaluate_perception)."""
    return 0.04 + 0.012 * dist ** 2


class GateMap:
    def __init__(self, prior_gates, prior_sigma=None, gate_threshold=16.0):
        sig = np.array(prior_sigma if prior_sigma is not None else CourseParams().prior_sigma)
        self.est = np.array([g.center for g in prior_gates], float)
        self.yaw = np.array([g.yaw for g in prior_gates], float)
        self.P = np.array([np.diag(sig ** 2) for _ in prior_gates])
        self.n_updates = np.zeros(len(prior_gates), int)
        self.thr = gate_threshold

    def update(self, z, dist, first_active=0):
        """Fuse one world-frame gate-centre measurement. Returns the gate index or None."""
        R = np.eye(3) * measurement_sigma(dist) ** 2
        best, best_d2 = None, self.thr
        for k in range(first_active, len(self.est)):
            e = z - self.est[k]
            S = self.P[k] + R
            d2 = float(e @ np.linalg.solve(S, e))
            if d2 < best_d2:
                best, best_d2 = k, d2
        if best is None:
            return None
        S = self.P[best] + R
        K = self.P[best] @ np.linalg.inv(S)
        self.est[best] = self.est[best] + K @ (z - self.est[best])
        self.P[best] = (np.eye(3) - K) @ self.P[best]
        self.n_updates[best] += 1
        return best

    def sigma(self, k):
        return np.sqrt(np.diag(self.P[k]))
