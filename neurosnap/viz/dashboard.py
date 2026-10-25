"""OpenCV dashboard: top-down map, FPV camera with detections, and a flight HUD."""
import cv2
import numpy as np

MAP = 480
CAM_W, CAM_H = 384, 288
W = MAP + CAM_W + 24
H = MAP

ORANGE = (0, 140, 255)
CYAN = (230, 200, 40)
GREEN = (90, 220, 90)
GREY = (150, 150, 150)
WHITE = (240, 240, 240)
RED = (60, 60, 230)


class Dashboard:
    def __init__(self, course, title=''):
        self.course = course
        pts = np.vstack([course.start, course.finish] + [g.center for g in course.gates])
        lo, hi = pts[:, :2].min(0) - 1.5, pts[:, :2].max(0) + 1.5
        self.scale = (MAP - 40) / max(hi - lo)
        self.lo = lo - ((MAP - 40) / self.scale - (hi - lo)) / 2
        self.path = []
        self.speeds = []
        self.title = title

    def w2m(self, xy):
        p = (np.asarray(xy)[..., :2] - self.lo) * self.scale + 20
        return np.stack([p[..., 0], MAP - p[..., 1]], -1)

    def _pt(self, xy):
        return tuple(int(v) for v in np.round(self.w2m(xy)))

    def draw(self, mission, t, img, dets, traj, nxt):
        q = mission.quad
        self.path.append(q.p.copy())
        self.speeds.append(float(np.linalg.norm(q.v)))
        canvas = np.full((H, W, 3), 28, np.uint8)
        m = canvas[:, :MAP]
        cv2.rectangle(m, (0, 0), (MAP - 1, MAP - 1), (60, 60, 60), 1)
        # true gates
        for i, g in enumerate(self.course.gates):
            d = 0.55 * np.array([-np.sin(g.yaw), np.cos(g.yaw)])
            col = (80, 80, 80) if i < nxt else ORANGE
            cv2.line(m, self._pt(g.center[:2] - d), self._pt(g.center[:2] + d), col, 4, cv2.LINE_AA)
            cv2.putText(m, str(i + 1), self._pt(g.center[:2] + 1.6 * d), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        # prior map and current estimates (1-sigma circles)
        for i, g in enumerate(mission.prior):
            c = self._pt(g.center)
            cv2.drawMarker(m, c, GREY, cv2.MARKER_TILTED_CROSS, 8, 1)
        for i in range(nxt, len(self.course.gates)):
            c = self._pt(mission.gmap.est[i])
            r = max(2, int(mission.gmap.sigma(i)[:2].mean() * self.scale))
            cv2.circle(m, c, r, CYAN, 1, cv2.LINE_AA)
            cv2.circle(m, c, 2, CYAN, -1)
        # current plan
        _, ps, _, _ = traj.sample(0.05)
        ps = ps[np.searchsorted(np.arange(len(ps)) * 0.05 + traj.t0, t):]
        if len(ps) > 1:
            cv2.polylines(m, [np.round(self.w2m(ps)).astype(np.int32)], False, GREEN, 1, cv2.LINE_AA)
        if len(self.path) > 1:
            cv2.polylines(m, [np.round(self.w2m(np.array(self.path))).astype(np.int32)], False, WHITE, 2,
                          cv2.LINE_AA)
        cv2.circle(m, self._pt(self.course.finish), 6, RED, 2)
        # drone
        yaw = np.arctan2(q.R[1, 0], q.R[0, 0])
        c = self.w2m(q.p)
        tri = np.array([[9, 0], [-6, 5], [-6, -5]], float)
        rot = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        cv2.fillConvexPoly(m, np.round(tri @ rot.T + c).astype(np.int32), RED, cv2.LINE_AA)
        cv2.putText(m, 'top view  (orange: gates, x: prior, cyan: estimate, green: plan)', (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1, cv2.LINE_AA)
        # FPV
        if img is not None:
            vis = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
            sx = CAM_W / img.shape[1]
            for d in dets:
                pts = np.round(d['corners_uv'] * sx).astype(np.int32)
                cv2.polylines(vis, [pts], True, GREEN, 2, cv2.LINE_AA)
                cc = tuple(int(v) for v in d['center_uv'] * sx)
                cv2.putText(vis, f'{np.linalg.norm(d["t_cam"]):.1f} m  {d["score"]:.2f}', (cc[0] + 5, cc[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
            canvas[8:8 + CAM_H, MAP + 12:MAP + 12 + CAM_W] = vis
        # HUD
        x0, y0 = MAP + 14, CAM_H + 30
        n = len(self.course.gates)
        lines = [self.title,
                 f't = {t:5.1f} s    speed = {self.speeds[-1]:4.2f} m/s    alt = {q.p[2]:4.2f} m',
                 f'gates passed {nxt}/{n}    detections {len(dets)}',
                 f'planner {mission.planner.name}: {mission.planner.times_ms[-1]:5.2f} ms',
                 f'perception: {getattr(mission.perception, "last_ms", 0.0):5.1f} ms (GateNet + PnP)']
        for i, s in enumerate(lines):
            cv2.putText(canvas, s, (x0, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        ORANGE if i == 0 else WHITE, 1, cv2.LINE_AA)
        sp = np.array(self.speeds[-150:])
        if len(sp) > 1:
            gx = x0 + np.linspace(0, CAM_W - 10, len(sp))
            gy = H - 12 - sp / 3.5 * 50
            cv2.polylines(canvas, [np.stack([gx, gy], 1).astype(np.int32)], False, CYAN, 1, cv2.LINE_AA)
            cv2.putText(canvas, 'speed', (x0, H - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
        return canvas
