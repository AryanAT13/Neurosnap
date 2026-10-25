"""Stability and control analysis of the cascaded PID on the Crazyflie model.

1. Closed-loop step responses on the full nonlinear model: roll / pitch 10 deg, yaw 45 deg,
   altitude +0.5 m, and rejection of a horizontal gust (0.05 N, ~19% of weight, for 0.5 s).
2. Linear analysis of the roll axis about hover: rate loop and attitude loop open-loop
   transfer functions (PI rate controller, motor lag, 1-sample delay as a Pade term), Bode
   gain / phase margins, and closed-loop poles as the attitude gain is swept.

    python scripts/control_analysis.py
Outputs results/control_analysis.json, docs/figures/control_steps.png, control_bode.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import FIG_DIR, RESULTS_DIR, Config  # noqa: E402
from neurosnap.geometry import quat_from_euler, quat_to_rot, rot_to_euler  # noqa: E402
from neurosnap.sim.controller import CascadedPID  # noqa: E402
from neurosnap.sim.quadrotor import Quadrotor  # noqa: E402

DT = 1e-3
T_STEP = 0.5


def simulate(kind, t_end=4.0):
    cfg = Config()
    q = Quadrotor(cfg.quad)
    q.reset(np.array([0.0, 0.0, 1.0]))
    c = CascadedPID(q, cfg.ctrl)
    ts, ys, rs = [], [], []
    rpm = np.full(4, cfg.quad.hover_rpm)
    for k in range(int(t_end / DT)):
        t = k * DT
        on = t >= T_STEP
        ref = dict(p=np.array([0.0, 0.0, 1.0]), v=np.zeros(3), a=np.zeros(3), yaw=0.0)
        if kind in ('roll', 'pitch', 'yaw'):
            ang = dict(roll=10.0, pitch=10.0, yaw=45.0)[kind] if on else 0.0
            e = [0.0, 0.0, 0.0]
            e[['roll', 'pitch', 'yaw'].index(kind)] = np.radians(ang)
            ref['Rd'] = quat_to_rot(quat_from_euler(*e))
            r = ang
        elif kind == 'altitude':
            ref['p'] = np.array([0.0, 0.0, 1.5 if on else 1.0])
            r = ref['p'][2]
        else:                                           # gust
            q.ext_force = np.array([0.05, 0.0, 0.0]) if T_STEP <= t < T_STEP + 0.5 else np.zeros(3)
            r = 0.0
        if k % 2 == 0:
            rpm = c.compute(q.p, q.v, q.R, q.w, ref, 2 * DT)
        q.step(rpm, DT)
        if kind in ('roll', 'pitch', 'yaw'):
            y = np.degrees(rot_to_euler(q.R)[['roll', 'pitch', 'yaw'].index(kind)])
        elif kind == 'altitude':
            y = q.p[2]
        else:
            y = q.p[0]
        ts.append(t)
        ys.append(y)
        rs.append(r)
    return np.array(ts), np.array(ys), np.array(rs)


def step_metrics(t, y, y0, y1):
    m = t >= T_STEP
    t, y = t[m] - T_STEP, (y[m] - y0) / (y1 - y0)
    t10 = t[np.argmax(y >= 0.1)]
    t90 = t[np.argmax(y >= 0.9)]
    over = max(0.0, (y.max() - 1.0) * 100)
    outside = np.where(np.abs(y - 1.0) > 0.05)[0]
    settle = t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(t) else 0.0
    return dict(rise_s=float(t90 - t10), overshoot_pct=float(over), settle_5pct_s=float(settle))


def roll_loops(kp_att=None):
    cp = Config().ctrl
    tau = Config().quad.motor_tau
    Td = 2.0 / cp.rate_hz                              # one control period of delay
    kp_r, ki_r = cp.kp_rate[0], cp.ki_rate[0]
    kp_a = cp.kp_att[0] if kp_att is None else kp_att
    pade_n, pade_d = np.array([-Td / 2, 1.0]), np.array([Td / 2, 1.0])
    # rate loop: (kp s + ki)/s * 1/(tau s + 1) * pade * 1/s
    Ln = np.polymul([kp_r, ki_r], pade_n)
    Ld = np.polymul(np.polymul([1.0, 0.0, 0.0], [tau, 1.0]), pade_d)
    # closed rate loop, then attitude loop kp_a * T_r / s
    Tn, Td_ = Ln, np.polyadd(Ld, Ln)
    An = kp_a * Tn
    Ad = np.polymul(Td_, [1.0, 0.0])
    return (Ln, Ld), (An, Ad)


def margins(num, den, w):
    s = 1j * w
    L = np.polyval(num, s) / np.polyval(den, s)
    mag, ph = np.abs(L), np.unwrap(np.angle(L))
    i = np.argmax(mag < 1.0)
    pm = 180 + np.degrees(ph[i])
    j = np.where(np.diff(np.sign(ph + np.pi)))[0]
    gm = float(-20 * np.log10(mag[j[0]])) if len(j) else float('inf')
    return L, float(pm), gm, float(w[i])


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = {}
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.4))
    for i, (kind, title, y0, y1, unit) in enumerate([
            ('roll', 'Roll 10 deg', 0, 10, 'deg'), ('pitch', 'Pitch 10 deg', 0, 10, 'deg'),
            ('yaw', 'Yaw 45 deg', 0, 45, 'deg'), ('altitude', 'Altitude +0.5 m', 1.0, 1.5, 'm'),
            ('gust', 'Gust 0.05 N for 0.5 s (x)', 0, 0, 'm')]):
        t, y, r = simulate(kind)
        ax[i].plot(t, r, 'k--', lw=0.8, label='reference')
        ax[i].plot(t, y, color='#3c6fb0', label='response')
        ax[i].set_title(title)
        ax[i].set_xlabel('t (s)')
        ax[i].set_ylabel(unit)
        ax[i].grid(alpha=0.3)
        if kind == 'gust':
            peak = float(np.abs(y).max())
            after = np.where((t > T_STEP + 0.5) & (np.abs(y) > 0.02))[0]
            rec = float(t[after[-1]] - (T_STEP + 0.5)) if len(after) else 0.0
            out[kind] = dict(peak_dev_m=peak, recover_2cm_s=rec)
        else:
            out[kind] = step_metrics(t, y, y0, y1)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'control_steps.png'), dpi=130)

    w = np.logspace(-1, 3.5, 4000)
    (Ln, Ld), (An, Ad) = roll_loops()
    Lr, pm_r, gm_r, wc_r = margins(Ln, Ld, w)
    La, pm_a, gm_a, wc_a = margins(An, Ad, w)
    gains = np.linspace(0.5, 200, 400)
    poles, unstable_at = [], None
    for k in gains:
        _, (An_k, Ad_k) = roll_loops(k)
        pk = np.roots(np.polyadd(Ad_k, An_k))
        poles.append(pk)
        if unstable_at is None and np.any(pk.real > 0):
            unstable_at = float(k)
    _, (An0, Ad0) = roll_loops()
    design = np.roots(np.polyadd(Ad0, An0))
    out['roll_rate_loop'] = dict(phase_margin_deg=pm_r, gain_margin_db=gm_r, crossover_rad_s=wc_r)
    out['roll_attitude_loop'] = dict(phase_margin_deg=pm_a, gain_margin_db=gm_a, crossover_rad_s=wc_a)
    out['attitude_gain'] = dict(design=Config().ctrl.kp_att[0], unstable_above=unstable_at,
                                design_poles=[[float(p.real), float(p.imag)] for p in design])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for L, lab in [(Lr, f'rate loop (PM {pm_r:.0f} deg)'), (La, f'attitude loop (PM {pm_a:.0f} deg)')]:
        ax[0].semilogx(w, 20 * np.log10(np.abs(L)), label=lab)
        ax[1].semilogx(w, np.degrees(np.unwrap(np.angle(L))), label=lab)
    ax[0].axhline(0, color='k', lw=0.6)
    ax[1].axhline(-180, color='k', lw=0.6)
    ax[0].set_ylabel('|L| (dB)')
    ax[1].set_ylabel('phase (deg)')
    for a in ax[:2]:
        a.set_xlabel('frequency (rad/s)')
        a.grid(alpha=0.3)
        a.legend()
    ax[0].set_title('Roll axis open-loop gain')
    ax[1].set_title('Roll axis open-loop phase')
    P = np.array(poles)
    sc = ax[2].scatter(P.real.ravel(), P.imag.ravel(), c=np.repeat(gains, P.shape[1]), s=3, cmap='viridis')
    ax[2].scatter(design.real, design.imag, marker='x', color='red', s=60, label=f'design (K_att={Config().ctrl.kp_att[0]:.0f})')
    ax[2].axvline(0, color='k', lw=0.6)
    ax[2].set_xlim(-200, 50)
    ax[2].set_xlabel('Re')
    ax[2].set_ylabel('Im')
    ax[2].set_title('Closed-loop roll poles vs attitude gain 0.5..200')
    ax[2].legend()
    fig.colorbar(sc, ax=ax[2], label='K_att')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'control_bode.png'), dpi=130)
    with open(os.path.join(RESULTS_DIR, 'control_analysis.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
