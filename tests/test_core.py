import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neurosnap.config import CameraParams, Config, GateParams, QuadParams, TrajLimits  # noqa: E402
from neurosnap.geometry import quat_from_euler, quat_to_rot, rot_to_quat, rot_z  # noqa: E402
from neurosnap.traj.minsnap import cost_and_grad, heuristic_times, solve_fixed_times  # noqa: E402
from neurosnap.traj.poly import PiecewisePoly  # noqa: E402
from neurosnap.world.course import Gate, gate_crossing, sample_course  # noqa: E402

P = np.array([[0, 0, 1], [3, 1, 1.5], [6, -1, 1.2], [8, 2, 1.0]], float)
V0 = np.array([0.5, 0.2, 0.0])
A0 = np.array([0.0, 0.3, 0.1])


def test_poly_interpolates_and_is_c3():
    T = np.array([1.5, 2.0, 1.7])
    knots, _ = solve_fixed_times(P, T, V0, A0)
    tr = PiecewisePoly(T, knots)
    assert np.allclose(tr.eval(tr.tk, 0), P, atol=1e-9)
    assert np.allclose(tr.state(0.0)[1], V0) and np.allclose(tr.state(0.0)[2], A0)
    for t in tr.tk[1:-1]:
        for d in range(4):
            assert np.allclose(tr.eval(t - 1e-7, d), tr.eval(t + 1e-7, d), atol=1e-4)
    assert np.allclose(tr.state(tr.duration)[1], 0, atol=1e-9)


def test_snap_cost_matches_numerical_integral():
    T = np.array([1.5, 2.0, 1.7])
    knots, J = solve_fixed_times(P, T, V0, A0)
    tr = PiecewisePoly(T, knots)
    ts = np.linspace(0, tr.duration, 200001)
    snap = tr.eval(ts, 4)
    J_num = np.trapezoid(np.sum(snap ** 2, axis=1), ts)
    assert abs(J - J_num) / J < 1e-3
    assert abs(tr.snap_cost() - J) / J < 1e-9


def test_qp_solution_is_optimal():
    T = np.array([1.5, 2.0, 1.7])
    knots, J = solve_fixed_times(P, T, V0, A0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        k = knots.copy()
        k[1:-1, 1:] += rng.normal(0, 0.05, k[1:-1, 1:].shape)
        assert PiecewisePoly(T, k).snap_cost() > J


def test_time_gradient_envelope():
    T = heuristic_times(P, TrajLimits())
    J, g, _ = cost_and_grad(P, T, V0, A0)
    for s in range(len(T)):
        e = np.zeros_like(T)
        e[s] = 1e-6
        num = (cost_and_grad(P, T + e, V0, A0)[0] - cost_and_grad(P, T - e, V0, A0)[0]) / 2e-6
        assert abs(num - g[s]) < 1e-3 * max(1.0, abs(g[s]))


def test_time_scaling_keeps_start_state():
    T = np.array([1.5, 2.0, 1.7])
    tr = PiecewisePoly(T, solve_fixed_times(P, T, V0, A0)[0]).time_scaled(1.5)
    assert np.allclose(tr.state(0.0)[1], V0) and np.allclose(tr.eval(tr.tk, 0), P)


def test_minsnap_rotation_invariant():
    T = np.array([1.5, 2.0, 1.7])
    R = rot_z(0.7)
    _, J1 = solve_fixed_times(P, T, V0, A0)
    _, J2 = solve_fixed_times(P @ R.T, T, R @ V0, R @ A0)
    assert abs(J1 - J2) < 1e-8 * J1


def test_quaternion_roundtrip():
    q = quat_from_euler(0.2, -0.3, 1.1)
    assert np.allclose(rot_to_quat(quat_to_rot(q)), q) or np.allclose(rot_to_quat(quat_to_rot(q)), -q)


def test_hover_and_controller():
    from neurosnap.sim.controller import CascadedPID
    from neurosnap.sim.quadrotor import Quadrotor
    cfg = Config()
    q = Quadrotor(cfg.quad)
    q.reset(np.array([0, 0, 1.0]))
    for _ in range(500):
        q.step(np.full(4, cfg.quad.hover_rpm), 1e-3)
    assert abs(q.p[2] - 1.0) < 1e-3
    ctrl = CascadedPID(q, cfg.ctrl)
    ref = dict(p=np.array([0.5, -0.3, 1.4]), v=np.zeros(3), a=np.zeros(3), yaw=0.5)
    for k in range(4000):
        if k % 2 == 0:
            rpm = ctrl.compute(q.p, q.v, q.R, q.w, ref, 2e-3)
        q.step(rpm, 1e-3)
    assert np.linalg.norm(q.p - ref['p']) < 0.02


def test_thrust_to_weight():
    qp = QuadParams()
    assert abs(4 * qp.kf * qp.max_rpm ** 2 / (qp.mass * 9.81) - qp.thrust2weight) < 1e-9


def test_gate_crossing():
    gp = GateParams()
    g = Gate(np.array([2.0, 0.0, 1.0]), 0.0)
    assert gate_crossing(np.array([1.9, 0.0, 1.0]), np.array([2.1, 0.05, 1.0]), g, gp, 0.06) == 'pass'
    assert gate_crossing(np.array([1.9, 0.45, 1.0]), np.array([2.1, 0.45, 1.0]), g, gp, 0.06) == 'crash'
    assert gate_crossing(np.array([1.9, 2.0, 1.0]), np.array([2.1, 2.0, 1.0]), g, gp, 0.06) is None
    assert gate_crossing(np.array([1.0, 0.0, 1.0]), np.array([1.5, 0.0, 1.0]), g, gp, 0.06) is None


def test_renderer_labels_and_pnp():
    from neurosnap.perception.detector import solve_gate_pnp
    from neurosnap.vision.renderer import Renderer, sample_style
    rng = np.random.default_rng(1)
    course = sample_course(rng)
    r = Renderer()
    g = course.gates[0]
    p = g.center - 3.0 * g.normal
    img, labels = r.render(course, p, rot_z(g.yaw), sample_style(rng))
    assert img.shape == (CameraParams().height, CameraParams().width, 3)
    lab = [l for l in labels if l['gate'] == 0][0]
    assert np.allclose(lab['center_uv'], [CameraParams().width / 2, lab['center_uv'][1]], atol=1.0)
    t, n = solve_gate_pnp(lab['corners_uv'], CameraParams().K, GateParams().inner)
    assert np.linalg.norm(t - lab['center_cam']) < 1e-6
    tl, tr = lab['corners_uv'][0], lab['corners_uv'][1]
    assert tl[0] < tr[0]                       # TL is left of TR in the image


def test_neural_decode_hard_constraints():
    torch = pytest.importorskip('torch')
    from neurosnap.planning.neural import TransformerPlanner, decode, make_features
    from neurosnap.traj.dataset import MAX_KNOTS
    B = 4
    Pp = torch.randn(B, MAX_KNOTS, 3)
    mask = torch.arange(MAX_KNOTS)[None] <= torch.tensor([2, 4, 6, 8])[:, None]
    v0, a0 = torch.randn(B, 3), torch.randn(B, 3)
    out = TransformerPlanner()(make_features(Pp, v0, a0, mask), mask)
    T, X = decode(out, Pp, v0, a0, mask)
    assert torch.all(T > 0)
    assert torch.allclose(X[:, :, 0], Pp)
    assert torch.allclose(X[:, 0, 1], v0) and torch.allclose(X[:, 0, 2], a0)
    for b, n in enumerate([2, 4, 6, 8]):
        assert torch.all(X[b, n, 1:] == 0)


def test_plan_verifier_rejects_overshoot():
    from neurosnap.mission import accept_plan
    Pw = np.array([[0, 0, 1], [1.0, 0, 1], [4, 0, 1]], float)
    v0 = np.array([2.0, 0, 0])
    good = PiecewisePoly([0.5, 1.5], solve_fixed_times(Pw, [0.5, 1.5], v0, np.zeros(3))[0])
    slow = PiecewisePoly([3.0, 1.5], solve_fixed_times(Pw, [3.0, 1.5], v0, np.zeros(3))[0])
    assert accept_plan(good, good, 0.0, Pw[1])
    assert not accept_plan(slow, good, 0.0, Pw[1])     # 2 m/s into a 1 m, 3 s segment must reverse


def test_receding_rollout_teacher_finishes():
    from neurosnap.planning.planners import MinSnapPlanner
    from neurosnap.traj.dagger import receding_rollout
    t_fin, t_first, vis = receding_rollout(MinSnapPlanner(), np.vstack([P[:1], P[1:]]), collect=True)
    assert t_first < t_fin < 3 * t_first and len(vis) > 0
