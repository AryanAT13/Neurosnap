"""Closed-loop gate-course mission: perception -> gate map -> (re-)planning -> PID -> quadrotor.

perception modes
    blind   fly the noisy prior map only
    cnn     GateNet + PnP detections fused into the gate map (proposed)
    oracle  true gate positions (upper bound)
"""
import time

import numpy as np

from .config import Config
from .geometry import wrap_angle
from .perception.gatemap import GateMap
from .sim.controller import CascadedPID
from .sim.quadrotor import Quadrotor
from .vision.renderer import Renderer, camera_pose, default_style
from .world.course import gate_crossing, noisy_prior

COMMIT_DIST = 0.7          # stop re-planning when the next gate is this close
MAX_FUSE_RANGE = 7.0       # PnP fixes beyond this range are too noisy to fuse


def accept_plan(new, old, t, gate):
    """Cheap analytic certificate for a re-planned trajectory (applied to every planner).

    A mid-flight plan is rejected, and the current one kept, if
      * it reverses direction along its first segment (overshoot-and-return loop), or
      * it reaches the next gate much later than the plan it would replace.
    """
    if not hasattr(new, 'T'):
        return True
    T0 = float(new.T[0])
    d = new.knots[1, 0] - new.knots[0, 0]
    d = d / max(np.linalg.norm(d), 1e-6)
    ts = t + np.linspace(0.0, T0, 25)
    if np.min(new.eval(ts, 1) @ d) < -0.05:
        return False
    ts_old = np.linspace(t, old.t_end, 200)
    if hasattr(old, 'eval'):
        p_old = old.eval(ts_old, 0)
    else:
        p_old = np.array([old.state(x)[0] for x in ts_old])
    t_gate_old = ts_old[int(np.argmin(np.linalg.norm(p_old - gate, axis=1)))] - t
    return T0 <= 1.4 * t_gate_old + 0.15
FINISH_TOL = 0.35


class Mission:
    def __init__(self, course, planner, perception_mode='cnn', perception=None, residual=None,
                 cfg=None, seed=0, style=None, on_frame=None, wind_sigma=0.3, verify=True):
        self.cfg = cfg or Config()
        self.course = course
        self.planner = planner
        self.mode = perception_mode
        self.perception = perception
        self.rng = np.random.default_rng(seed)
        wind = np.concatenate([self.rng.normal(0, wind_sigma, 2), [0.0]])
        self.quad = Quadrotor(self.cfg.quad, wind=wind)
        self.ctrl = CascadedPID(self.quad, self.cfg.ctrl, residual)
        self.prior = noisy_prior(course, self.rng)
        if perception_mode == 'oracle':
            self.gmap = GateMap(course.gates, prior_sigma=(1e-3, 1e-3, 1e-3))
        else:
            self.gmap = GateMap(self.prior)
        self.renderer = Renderer(self.cfg.cam, self.cfg.gate, seed=seed) if perception_mode == 'cnn' or on_frame else None
        self.style = style or default_style()
        self.on_frame = on_frame
        self.verify = verify

    # ------------------------------------------------------------------ helpers
    def _waypoints(self, p_ref, nxt):
        return np.vstack([p_ref, self.gmap.est[nxt:], self.course.finish])

    def _perceive(self, p_est, R_est, nxt):
        q = self.quad
        img, _ = self.renderer.render(self.course, q.p, q.R, self.style, with_labels=False)
        dets = self.perception.detect(img) if self.perception is not None else []
        centre, R_wc = camera_pose(p_est, R_est, self.cfg.cam)
        used = []
        for d in dets:
            rng = float(np.linalg.norm(d['t_cam']))
            if rng > MAX_FUSE_RANGE:
                continue
            z = centre + R_wc @ d['t_cam']
            used.append(self.gmap.update(z, rng, first_active=nxt))
        return img, dets, used

    # ------------------------------------------------------------------ main loop
    def run(self, record=False):
        cfg, q, course = self.cfg, self.quad, self.course
        dt = 1.0 / cfg.sim_hz
        ctrl_every = int(round(cfg.sim_hz / cfg.ctrl.rate_hz))
        cam_every = int(round(cfg.sim_hz / cfg.cam.rate_hz))
        plan_every = int(round(cfg.sim_hz / cfg.replan_hz))
        first = course.gates[0].center - course.start
        yaw_ref = float(np.arctan2(first[1], first[0]))
        q.reset(course.start, yaw_ref)
        nxt = 0
        outcome = None
        pass_err, est_err, perc_ms, track_err = [], [], [], []
        log = dict(t=[], p=[], ref=[], v=[], plans=[]) if record else None
        if self.mode == 'cnn' and self.perception is not None:
            self._perceive(q.p, q.R, 0)
        traj = self.planner.plan(self._waypoints(course.start, 0), np.zeros(3), np.zeros(3), 0.0)
        n_plans, n_rejected = 1, 0
        if record:
            log['plans'].append((0.0, traj))
        rpm = np.full(4, cfg.quad.hover_rpm)
        t_finish = None
        steps = int(cfg.max_mission_time * cfg.sim_hz)
        img, dets = None, []
        for k in range(steps):
            t = k * dt
            # ---------------- perception
            p_est = q.p + self.rng.normal(0, 0.01, 3)
            v_est = q.v + self.rng.normal(0, 0.03, 3)
            if k % cam_every == 0 and k > 0 and (self.mode == 'cnn' or self.on_frame):
                if self.mode == 'cnn':
                    img, dets, _ = self._perceive(p_est, q.R, nxt)
                    perc_ms.append(self.perception.last_ms)
                else:
                    img, _ = self.renderer.render(course, q.p, q.R, self.style, with_labels=False)
                if self.on_frame:
                    self.on_frame(self, t, img, dets, traj, nxt)
            # ---------------- re-planning (receding horizon from the reference state)
            if k % plan_every == 0 and k > 0 and nxt < len(course.gates) and self.mode != 'oracle':
                p_r, v_r, a_r, _ = traj.state(t)
                if np.linalg.norm(self.gmap.est[nxt] - p_r) > COMMIT_DIST:
                    new = self.planner.plan(self._waypoints(p_r, nxt), v_r, a_r, t)
                    n_plans += 1
                    if self.verify and not accept_plan(new, traj, t, self.gmap.est[nxt]):
                        n_rejected += 1
                    else:
                        traj = new
                        if record:
                            log['plans'].append((t, traj))
            # ---------------- control
            p_r, v_r, a_r, _ = traj.state(t)
            if k % ctrl_every == 0:
                tgt = self.gmap.est[nxt] if nxt < len(course.gates) else course.finish
                to = tgt - q.p
                if np.linalg.norm(to[:2]) > 0.8:
                    desired = float(np.arctan2(to[1], to[0]))
                    yaw_ref += float(np.clip(wrap_angle(desired - yaw_ref), -3.0 * dt * ctrl_every,
                                             3.0 * dt * ctrl_every))
                ref = dict(p=p_r, v=v_r, a=a_r, yaw=yaw_ref)
                rpm = self.ctrl.compute(p_est, v_est, q.R, q.w, ref, dt * ctrl_every)
            p_prev = q.p.copy()
            q.step(rpm, dt)
            track_err.append(np.linalg.norm(q.p - p_r))
            if record and k % 10 == 0:
                log['t'].append(t)
                log['p'].append(q.p.copy())
                log['ref'].append(p_r.copy())
                log['v'].append(q.v.copy())
            # ---------------- events
            for gi, g in enumerate(course.gates):
                if np.linalg.norm(q.p - g.center) > 1.5:
                    continue
                res = gate_crossing(p_prev, q.p, g, cfg.gate, cfg.quad.radius)
                if res == 'crash':
                    outcome = 'crash'
                elif res == 'pass' and gi == nxt:
                    off = g.R.T @ (q.p - g.center)
                    pass_err.append(float(np.hypot(off[1], off[2])))
                    est_err.append(float(np.linalg.norm(self.gmap.est[gi] - g.center)))
                    nxt += 1
            if nxt < len(course.gates):
                g = course.gates[nxt]
                s0, s1 = np.dot(p_prev - g.center, g.normal), np.dot(q.p - g.center, g.normal)
                if s0 < 0 <= s1 and np.linalg.norm(q.p - g.center) < 4.0:
                    off = g.R.T @ (q.p - g.center)
                    if max(abs(off[1]), abs(off[2])) > cfg.gate.outer / 2 + cfg.quad.radius:
                        outcome = 'missed'
            if t > 0.5 and q.p[2] < 0.03:
                outcome = 'crash'
            if (nxt == len(course.gates) and np.linalg.norm(q.p - course.finish) < FINISH_TOL
                    and outcome is None):
                outcome, t_finish = 'success', t
            if outcome is not None:
                break
        if outcome is None:
            outcome = 'timeout'
        pt = np.array(self.planner.times_ms) if self.planner.times_ms else np.zeros(1)
        res = dict(outcome=outcome, success=outcome == 'success', gates_passed=nxt,
                   n_gates=len(course.gates), lap_time=t_finish,
                   pass_offset=float(np.mean(pass_err)) if pass_err else None,
                   gate_est_err=float(np.mean(est_err)) if est_err else None,
                   prior_err=float(np.mean([np.linalg.norm(a.center - b.center)
                                            for a, b in zip(self.prior, course.gates)])),
                   track_rmse=float(np.sqrt(np.mean(np.square(track_err)))),
                   plan_ms_mean=float(pt.mean()), plan_ms_p95=float(np.percentile(pt, 95)),
                   n_plans=n_plans, n_rejected=n_rejected, perc_ms=float(np.mean(perc_ms)) if perc_ms else None,
                   flight_time=t)
        if record:
            res['log'] = {k: (np.array(v) if k != 'plans' else v) for k, v in log.items()}
        return res
