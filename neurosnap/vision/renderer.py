"""Lightweight FPV renderer for gate courses (numpy + OpenCV, no GPU / game engine).

* Ground and sky: per-pixel ray casting onto the plane z = 0 with a procedural
  texture, distance fog and a sky gradient.
* Gates, stands and distractor objects: boxes drawn as flat-shaded polygons with
  back-face culling, near-plane clipping and painter's-algorithm ordering.
* Domain randomisation of colours, textures, lighting, noise and blur per scene.

Every render also returns exact labels (projected opening corners, gate pose in the
camera frame), so perception data is self-labelled from the renderer.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import CameraParams, GateParams
from ..geometry import rot_y
from ..world.course import POLE_WIDTH

# body (x fwd, y left, z up) -> OpenCV camera (x right, y down, z forward)
_R_BODY_CAM0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
NEAR = 0.05


def camera_pose(p, R_wb, cam):
    """World pose of the camera: (centre, R_wc) with X_c = R_wc^T (X_w - centre)."""
    R_bc = rot_y(-np.radians(cam.tilt_up_deg)) @ _R_BODY_CAM0
    return p + R_wb @ np.array(cam.offset), R_wb @ R_bc


@dataclass
class SceneStyle:
    sky_top: np.ndarray
    sky_horizon: np.ndarray
    ground_tex: np.ndarray
    tex_scale: float
    fog_density: float
    gate_color: np.ndarray
    light_dir: np.ndarray
    ambient: float
    brightness: float
    contrast: float
    noise: float
    blur: int
    gain: np.ndarray


def _noise_texture(rng, size=64):
    base = rng.integers(0, 255, (3,))
    alt = np.clip(base + rng.integers(-90, 90, (3,)), 0, 255)
    noise = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), rng.uniform(1.0, 4.0))
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    kind = rng.integers(0, 3)
    if kind == 1:                     # checkerboard floor (gym / indoor arena look)
        ii, jj = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        noise = 0.7 * (((ii // (size // 4)) + (jj // (size // 4))) % 2) + 0.3 * noise
    elif kind == 2:                   # tiles / stripes
        ii = np.arange(size)[:, None] * np.ones((1, size))
        noise = 0.6 * ((ii // (size // 8)) % 2) + 0.4 * noise
    tex = base[None, None] * (1 - noise[..., None]) + alt[None, None] * noise[..., None]
    return tex.astype(np.uint8)


def sample_style(rng):
    hue = rng.random()
    # gate colour: mostly saturated "racing" colours, sometimes arbitrary
    if rng.random() < 0.7:
        hsv = np.uint8([[[int(hue * 180), int(rng.uniform(150, 255)), int(rng.uniform(150, 255))]]])
    else:
        hsv = np.uint8([[[int(hue * 180), int(rng.uniform(0, 255)), int(rng.uniform(60, 255))]]])
    gate_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(float)
    ld = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.4, 1.5)])
    return SceneStyle(
        sky_top=rng.uniform([120, 60, 20], [255, 200, 140]),
        sky_horizon=rng.uniform([150, 140, 120], [255, 255, 255]),
        ground_tex=_noise_texture(rng),
        tex_scale=rng.uniform(1.0, 6.0),
        fog_density=rng.uniform(0.0, 0.08),
        gate_color=gate_bgr,
        light_dir=ld / np.linalg.norm(ld),
        ambient=rng.uniform(0.35, 0.7),
        brightness=rng.uniform(-30, 30),
        contrast=rng.uniform(0.75, 1.25),
        noise=rng.uniform(0, 10),
        blur=int(rng.choice([0, 0, 3, 5])),
        gain=rng.uniform(0.85, 1.15, 3),
    )


DEFAULT_STYLE = None


def default_style():
    """Fixed, clean style used by the live demo."""
    global DEFAULT_STYLE
    if DEFAULT_STYLE is None:
        s = sample_style(np.random.default_rng(3))
        s.gate_color = np.array([0.0, 110.0, 255.0])
        s.sky_top = np.array([235.0, 180.0, 110.0])
        s.sky_horizon = np.array([250.0, 235.0, 215.0])
        s.fog_density, s.brightness, s.contrast, s.noise, s.blur = 0.02, 0.0, 1.0, 2.0, 0
        s.gain = np.ones(3)
        s.ambient = 0.55
        DEFAULT_STYLE = s
    return DEFAULT_STYLE


def _box_faces(center, R, half):
    """Six faces (4 vertices each, outward normal) of an oriented box."""
    c = np.asarray(center, float)
    out = []
    for axis in range(3):
        for sign in (-1, 1):
            n_local = np.zeros(3)
            n_local[axis] = sign
            u, w = [a for a in range(3) if a != axis]
            quad = []
            for su, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                v = np.zeros(3)
                v[axis] = sign * half[axis]
                v[u] = su * half[u]
                v[w] = sw * half[w]
                quad.append(c + R @ v)
            out.append((np.array(quad), R @ n_local))
    return out


def gate_boxes(gate, gp):
    """Frame bars + ground stand of one gate as (center, R, half_extents)."""
    R = gate.R
    hi, b, d = gp.inner / 2, gp.border, gp.depth
    ho = hi + b
    boxes = []
    for z in (hi + b / 2, -(hi + b / 2)):
        boxes.append((gate.center + R @ np.array([0, 0, z]), R, np.array([d / 2, ho, b / 2])))
    for y in (hi + b / 2, -(hi + b / 2)):
        boxes.append((gate.center + R @ np.array([0, y, 0]), R, np.array([d / 2, b / 2, hi])))
    bottom = gate.center[2] - ho
    if bottom > 0.02:
        boxes.append((np.array([gate.center[0], gate.center[1], bottom / 2]), R,
                      np.array([POLE_WIDTH / 2, POLE_WIDTH / 2, bottom / 2])))
    return boxes


def _clip_near(Pc):
    """Sutherland-Hodgman clip of a camera-frame polygon against z >= NEAR."""
    out = []
    n = len(Pc)
    for i in range(n):
        a, b = Pc[i], Pc[(i + 1) % n]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out)


class Renderer:
    def __init__(self, cam=None, gp=None, seed=None):
        self.cam = cam or CameraParams()
        self.rng = np.random.default_rng(seed)        # sensor noise
        self.gp = gp or GateParams()
        W, H = self.cam.width, self.cam.height
        self.K = self.cam.K
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        rays = np.stack([(uu - W / 2) / self.cam.fx, (vv - H / 2) / self.cam.fx, np.ones_like(uu)], -1)
        self.rays_c = rays.reshape(-1, 3)
        self._scene_key = None

    # ----------------------------------------------------------------- geometry
    def _build_scene(self, course, style):
        key = (id(course), id(style))
        if key == self._scene_key:
            return
        faces = []
        for gi, g in enumerate(course.gates):
            for bi, (c, R, h) in enumerate(gate_boxes(g, self.gp)):
                col = style.gate_color if bi < 4 else np.array([90.0, 90.0, 90.0])
                for quad, n in _box_faces(c, R, h):
                    faces.append((quad, n, col))
        for ob in course.obstacles:
            for quad, n in _box_faces(ob.center, np.eye(3), ob.size / 2):
                faces.append((quad, n, np.array(ob.color, float)))
        self._faces = faces
        self._scene_key = key

    def project(self, Xw, centre, R_wc):
        Xc = (np.atleast_2d(Xw) - centre) @ R_wc
        uv = (Xc[:, :2] / Xc[:, 2:3]) * self.cam.fx + np.array([self.cam.width / 2, self.cam.height / 2])
        return uv, Xc

    # ----------------------------------------------------------------- drawing
    def _background(self, centre, R_wc, st):
        W, H = self.cam.width, self.cam.height
        d = self.rays_c @ R_wc.T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        img = np.empty((d.shape[0], 3))
        elev = np.clip(d[:, 2], 0, 1)[:, None]
        sky = st.sky_horizon * (1 - elev ** 0.6) + st.sky_top * elev ** 0.6
        img[:] = sky
        g = d[:, 2] < -1e-4
        if np.any(g) and centre[2] > 0:
            t = -centre[2] / d[g, 2]
            hit = centre[None, :2] + t[:, None] * d[g, :2]
            th, tw = st.ground_tex.shape[:2]
            iu = (np.floor(hit[:, 0] / st.tex_scale * tw).astype(np.int64)) % tw
            iv = (np.floor(hit[:, 1] / st.tex_scale * th).astype(np.int64)) % th
            col = st.ground_tex[iv, iu].astype(float)
            fog = 1 - np.exp(-st.fog_density * t)[:, None]
            img[g] = col * (1 - fog) + st.sky_horizon * fog
        return img.reshape(H, W, 3)

    def render(self, course, p, R_wb, style, with_labels=True):
        cam = self.cam
        centre, R_wc = camera_pose(p, R_wb, cam)
        self._build_scene(course, style)
        img = self._background(centre, R_wc, style).astype(np.uint8)
        polys = []
        for quad, n, col in self._faces:
            fc = quad.mean(0)
            if np.dot(n, centre - fc) <= 0:
                continue
            Pc = (quad - centre) @ R_wc
            if Pc[:, 2].max() < NEAR or Pc[:, 2].min() > cam.max_range + 8:
                continue
            if Pc[:, 2].min() < NEAR:
                Pc = _clip_near(Pc)
                if len(Pc) < 3:
                    continue
            uv = Pc[:, :2] / Pc[:, 2:3] * cam.fx + np.array([cam.width / 2, cam.height / 2])
            if (uv[:, 0].max() < -2 or uv[:, 0].min() > cam.width + 2 or
                    uv[:, 1].max() < -2 or uv[:, 1].min() > cam.height + 2):
                continue
            shade = style.ambient + (1 - style.ambient) * max(0.0, float(np.dot(n, style.light_dir)))
            depth = float(np.linalg.norm(fc - centre))
            fog = 1 - np.exp(-style.fog_density * depth)
            c = col * shade * (1 - fog) + style.sky_horizon * fog
            polys.append((depth, uv, c))
        polys.sort(key=lambda x: -x[0])
        for _, uv, c in polys:
            pts = np.clip(np.round(uv * 16), -1e6, 1e6).astype(np.int32)
            cv2.fillConvexPoly(img, pts, tuple(float(v) for v in np.clip(c, 0, 255)), cv2.LINE_AA, 4)
        img = self._postprocess(img, style)
        labels = self.labels(course, centre, R_wc) if with_labels else None
        return img, labels

    def _postprocess(self, img, st):
        f = img.astype(np.float32)
        f = (f - 128) * st.contrast + 128 + st.brightness
        f *= st.gain[None, None, :]
        if st.blur:
            k = np.zeros((st.blur, st.blur), np.float32)
            k[st.blur // 2, :] = 1.0 / st.blur          # horizontal motion blur
            f = cv2.filter2D(f, -1, k)
        if st.noise > 0:
            f += self.rng.normal(0, st.noise, f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------- labels
    def labels(self, course, centre, R_wc):
        cam = self.cam
        out = []
        for gi, g in enumerate(course.gates):
            uv_c, Xc = self.project(g.center, centre, R_wc)
            z = Xc[0, 2]
            dist = float(np.linalg.norm(g.center - centre))
            if z < 0.3 or dist > cam.max_range:
                continue
            u, v = uv_c[0]
            if not (0 <= u < cam.width and 0 <= v < cam.height):
                continue
            view = (g.center - centre) / dist
            facing = float(np.dot(view, g.normal))
            if abs(facing) < 0.2:                   # seen almost edge-on
                continue
            corners = g.inner_corners(self.gp.inner)
            order = [0, 1, 2, 3] if facing > 0 else [1, 0, 3, 2]
            uv, _ = self.project(corners[order], centre, R_wc)
            out.append(dict(gate=gi, center_uv=uv_c[0], corners_uv=uv, center_cam=Xc[0],
                            dist=dist, front=facing > 0))
        return out
