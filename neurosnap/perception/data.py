"""Self-labelled, domain-randomised perception data rendered on the fly.

Camera poses are sampled the way they occur in flight: along the course between
consecutive gates, looking roughly at the next gate, with forward pitch and roll
from manoeuvring, plus a share of fully random views.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..geometry import quat_from_euler, quat_to_rot
from ..vision.renderer import Renderer, sample_style
from ..world.course import sample_course
from .net import CORNER_S, STRIDE

MAX_OBJ = 8
MEAN = np.array([0.45, 0.45, 0.45], np.float32)
STD = np.array([0.25, 0.25, 0.25], np.float32)


def to_tensor(img_bgr):
    x = (img_bgr[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def sample_view(rng, course):
    pts = np.vstack([course.start] + [g.center for g in course.gates])
    if rng.random() < 0.1:
        lo, hi = pts.min(0) - 2, pts.max(0) + 2
        p = rng.uniform(lo, hi)
        p[2] = rng.uniform(0.3, 2.5)
        yaw = rng.uniform(-np.pi, np.pi)
    else:
        k = int(rng.integers(0, len(pts) - 1))
        a, b = pts[k], pts[k + 1]
        s = rng.uniform(0.0, 0.95)
        d = b - a
        hd = np.arctan2(d[1], d[0])
        lateral = np.array([-np.sin(hd), np.cos(hd), 0.0])
        p = a + s * d + lateral * rng.normal(0, 0.4) + np.array([0, 0, rng.normal(0, 0.25)])
        p[2] = max(p[2], 0.15)
        to = b - p
        yaw = np.arctan2(to[1], to[0]) + rng.normal(0, np.radians(18))
    roll = rng.normal(0, np.radians(10))
    pitch = rng.normal(np.radians(8), np.radians(10))       # nose-down in forward flight
    R = quat_to_rot(quat_from_euler(roll, pitch, yaw))
    return p, R


def gaussian_radius(w, h):
    return max(0.8, 0.12 * min(w, h) / STRIDE + 0.5)


def encode_targets(labels, H, W):
    Ho, Wo = H // STRIDE, W // STRIDE
    hm = np.zeros((1, Ho, Wo), np.float32)
    ind = np.zeros(MAX_OBJ, np.int64)
    mask = np.zeros(MAX_OBJ, bool)
    off = np.zeros((MAX_OBJ, 2), np.float32)
    cor = np.zeros((MAX_OBJ, 8), np.float32)
    yy, xx = np.mgrid[0:Ho, 0:Wo]
    labels = sorted(labels, key=lambda l: l['dist'])[:MAX_OBJ]
    for k, lab in enumerate(labels):
        c = lab['center_uv'] / STRIDE
        ix, iy = int(np.clip(c[0], 0, Wo - 1)), int(np.clip(c[1], 0, Ho - 1))
        span = lab['corners_uv'].max(0) - lab['corners_uv'].min(0)
        sig = gaussian_radius(*span)
        g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sig ** 2))
        hm[0] = np.maximum(hm[0], g)
        hm[0, iy, ix] = 1.0
        ind[k] = iy * Wo + ix
        mask[k] = True
        off[k] = c - np.array([ix, iy])
        cor[k] = ((lab['corners_uv'] - lab['center_uv']) / CORNER_S).reshape(-1)
    return dict(hm=torch.from_numpy(hm), ind=torch.from_numpy(ind), mask=torch.from_numpy(mask),
                off=torch.from_numpy(off), cor=torch.from_numpy(cor))


def make_sample(rng, renderer):
    course = sample_course(rng)
    style = sample_style(rng)
    p, R = sample_view(rng, course)
    img, labels = renderer.render(course, p, R, style)
    return img, labels


class OnlineGateData(IterableDataset):
    """Infinite stream of freshly rendered, randomised training frames."""

    def __init__(self, seed=0, per_scene=4):
        self.seed = seed
        self.per_scene = per_scene

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid + int(torch.randint(0, 2 ** 31, (1,))))
        r = Renderer(seed=int(rng.integers(0, 2 ** 31)))
        while True:
            course = sample_course(rng)
            style = sample_style(rng)
            for _ in range(self.per_scene):
                p, R = sample_view(rng, course)
                img, labels = r.render(course, p, R, style)
                yield to_tensor(img), encode_targets(labels, *img.shape[:2])


class FixedGateData(Dataset):
    """Deterministic held-out set (frames + raw labels) for validation / evaluation."""

    def __init__(self, n, seed=12345):
        rng = np.random.default_rng(seed)
        r = Renderer(seed=seed)
        self.items = []
        for _ in range(n):
            img, labels = make_sample(rng, r)
            self.items.append((img, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, labels = self.items[i]
        return to_tensor(img), encode_targets(labels, *img.shape[:2])
