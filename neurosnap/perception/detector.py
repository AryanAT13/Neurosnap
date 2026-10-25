"""Run-time gate perception: GateNet inference, peak decoding and PnP pose recovery."""
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import CameraParams, GateParams
from .data import to_tensor
from .net import CORNER_S, STRIDE, GateNet


def decode(hm, off, cor, thr=0.35, k=8):
    """Network maps (single image, no batch) -> list of (score, centre_uv, corners_uv)."""
    p = torch.sigmoid(hm)
    keep = (F.max_pool2d(p, 3, 1, 1) == p) & (p > thr)
    ys, xs = torch.nonzero(keep[0], as_tuple=True)
    scores = p[0, ys, xs]
    order = torch.argsort(scores, descending=True)[:k]
    dets = []
    for i in order.tolist():
        y, x = int(ys[i]), int(xs[i])
        c = (np.array([x, y], float) + off[:, y, x].numpy()) * STRIDE
        corners = c + cor[:, y, x].numpy().reshape(4, 2) * CORNER_S
        dets.append((float(scores[i]), c, corners))
    return dets


def gate_object_points(size):
    h = size / 2
    # IPPE_SQUARE order: TL, TR, BR, BL in a frame with x right, y up
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def solve_gate_pnp(corners_uv, K, size):
    ok, rvec, tvec = cv2.solvePnP(gate_object_points(size), corners_uv.astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return tvec.reshape(3), R[:, 2]          # centre and plane normal in the camera frame


class GatePerception:
    def __init__(self, weights, cam=None, gp=None, thr=0.35, device='cpu'):
        self.cam = cam or CameraParams()
        self.gp = gp or GateParams()
        self.net = GateNet()
        ck = torch.load(weights, map_location='cpu', weights_only=False)
        self.net.load_state_dict(ck['state'])
        self.net.eval().to(device)
        self.device = device
        self.thr = thr
        self.K = self.cam.K
        self.last_ms = 0.0

    @torch.no_grad()
    def detect(self, img_bgr):
        t = time.perf_counter()
        x = to_tensor(img_bgr)[None].to(self.device)
        hm, off, cor = (o[0].cpu() for o in self.net(x))
        dets = []
        for score, c, corners in decode(hm, off, cor, self.thr):
            pose = solve_gate_pnp(corners, self.K, self.gp.inner)
            if pose is None or pose[0][2] <= 0.2:
                continue
            dets.append(dict(score=score, center_uv=c, corners_uv=corners, t_cam=pose[0], n_cam=pose[1]))
        self.last_ms = (time.perf_counter() - t) * 1e3
        return dets
