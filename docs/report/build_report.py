"""Build the project implementation report (PDF) from the saved results and figures.

    python docs/report/build_report.py [--github https://github.com/<user>/NeuroSnap]
Outputs NeuroSnap_Project_Report_Aryan_Amit_Tiwari_23BAI1409.pdf in the repository root.
Every table is read from results/*.json, so the report always matches the repository.
"""
import argparse
import json
import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG = os.path.join(ROOT, 'docs', 'figures')
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'NeuroSnap_Project_Report_Aryan_Amit_Tiwari_23BAI1409.pdf')

FONT_DIR = '/System/Library/Fonts/Supplemental'
pdfmetrics.registerFont(TTFont('Arial', os.path.join(FONT_DIR, 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', os.path.join(FONT_DIR, 'Arial Bold.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Italic', os.path.join(FONT_DIR, 'Arial Italic.ttf')))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', os.path.join(FONT_DIR, 'Arial Bold Italic.ttf')))
pdfmetrics.registerFont(TTFont('Mono', os.path.join(FONT_DIR, 'Courier New.ttf')))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic',
                              boldItalic='Arial-BoldItalic')

PAGE_W = A4[0] - 4.0 * cm
BLUE = '#1f4e9c'
base = ParagraphStyle('base', fontName='Arial', fontSize=10.3, leading=14.6, alignment=TA_JUSTIFY, spaceAfter=6)
title = ParagraphStyle('title', parent=base, fontName='Arial-Bold', fontSize=20, leading=25,
                       alignment=TA_CENTER, spaceAfter=10)
sub = ParagraphStyle('sub', parent=base, fontSize=12.5, alignment=TA_CENTER, spaceAfter=12)
centre = ParagraphStyle('centre', parent=base, alignment=TA_CENTER, spaceAfter=3)
h1 = ParagraphStyle('h1', parent=base, fontName='Arial-Bold', fontSize=14.5, leading=19, spaceBefore=12,
                    spaceAfter=2, alignment=0, keepWithNext=1)
h2 = ParagraphStyle('h2', parent=base, fontName='Arial-Bold', fontSize=11.5, leading=15, spaceBefore=8,
                    spaceAfter=4, alignment=0, keepWithNext=1)
cap = ParagraphStyle('cap', parent=base, fontSize=8.6, leading=11.5, alignment=TA_CENTER,
                     textColor=colors.HexColor('#444444'), spaceAfter=10)
cell = ParagraphStyle('cell', parent=base, fontSize=8.8, leading=11.6, alignment=0, spaceAfter=0)
cellb = ParagraphStyle('cellb', parent=cell, fontName='Arial-Bold')
cellr = ParagraphStyle('cellr', parent=cell, alignment=2)
eq = ParagraphStyle('eq', parent=base, fontName='Arial-Italic', alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
code = ParagraphStyle('code', fontName='Mono', fontSize=8.4, leading=11.2)
item = ParagraphStyle('item', parent=base, spaceAfter=2.5, alignment=0)

story = []


def P(text, style=base):
    story.append(Paragraph(text, style))


def rule():
    hr = HRFlowable(width='100%', thickness=0.9, color=colors.black, spaceBefore=1, spaceAfter=7)
    hr.keepWithNext = 1
    return hr


def H1(text):
    story.append(Paragraph(text, h1))
    story.append(rule())


def H2(text):
    story.append(Paragraph(text, h2))


def bullets(items, numbered=False):
    lf = ListFlowable([ListItem(Paragraph(t, item), leftIndent=16) for t in items],
                      bulletType='1' if numbered else 'bullet', start=None if numbered else '•',
                      leftIndent=16, bulletFontName='Arial', bulletFontSize=9.5)
    story.append(lf)
    story.append(Spacer(1, 4))


TRIM_DIR = os.path.join(os.path.dirname(__file__), '_build')


def img(name, width=PAGE_W):
    """Figure with its uniform white margins trimmed (matplotlib saves generous borders)."""
    src = PILImage.open(os.path.join(FIG, name)).convert('RGB')
    bg = PILImage.new('RGB', src.size, (255, 255, 255))
    from PIL import ImageChops
    box = ImageChops.difference(src, bg).convert('L').point(lambda v: 255 if v > 12 else 0).getbbox()
    if box:
        pad = 6
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, src.width), min(box[3] + pad, src.height))
        src = src.crop(box)
    os.makedirs(TRIM_DIR, exist_ok=True)
    path = os.path.join(TRIM_DIR, name)
    src.save(path)
    w, h = src.size
    return Image(path, width=width, height=width * h / w)


def figure(name, caption, width=PAGE_W):
    story.append(KeepTogether([img(name, width), Spacer(1, 4), Paragraph(caption, cap)]))


def figure_pair(a, cap_a, b, cap_b):
    w = PAGE_W / 2 - 6
    t = Table([[img(a, w), img(b, w)], [Paragraph(cap_a, cap), Paragraph(cap_b, cap)]], colWidths=[w + 6, w + 6])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 3),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    story.append(t)


def table(rows, widths, highlight=(), right_from=None, header=True):
    data = []
    for r, row in enumerate(rows):
        out = []
        for c, v in enumerate(row):
            st = cellb if (header and r == 0) or r in highlight else cell
            if right_from is not None and c >= right_from and r > 0:
                st = ParagraphStyle('x', parent=st, alignment=2)
            out.append(Paragraph(str(v), st))
        data.append(out)
    t = Table(data, colWidths=[PAGE_W * w for w in widths], repeatRows=1 if header else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9a9a9a')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
          ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8e8e8')))
    for r in highlight:
        st.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor('#eaf1fb')))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))


def note(text, color='#fdf6e3', border='#c9a227'):
    t = Table([[Paragraph(text, cell)]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(color)),
                           ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor(border)),
                           ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                           ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10)]))
    story.append(t)
    story.append(Spacer(1, 8))


def codeblock(text):
    t = Table([[Preformatted(text, code)]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f3f3')),
                           ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                           ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                           ('LEFTPADDING', (0, 0), (-1, -1), 9)]))
    story.append(t)
    story.append(Spacer(1, 8))


def link(url, text=None):
    return f'<a href="{url}" color="{BLUE}">{text or url}</a>'


def pct(x, d=1):
    v = f'{100 * x:.{d}f}'
    return (v[:-2] if v.endswith('.0') else v) + ' %'


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def build(github):
    bench = load('benchmark_summary.json')['summary']
    nb = load('benchmark_summary.json')['seeds']
    plan = load('planner_eval.json')
    perc = load('perception_eval.json')
    dag = load('dagger.json')
    ctrl = load('control_analysis.json')
    res = json.load(open(os.path.join(ROOT, 'models', 'residual_log.json')))
    n_missions = nb * len(bench)
    ns = bench['neurosnap']
    qp = bench['qp_opt+cnn']
    ps = plan['summary']
    teacher = ps['Min-snap QP, optimised T (teacher)']
    full = ps['Transformer, no snap loss, DAgger (full neural)']
    hyb = ps['Transformer, no snap loss, DAgger + QP derivatives (hybrid)']
    d0, d3 = dag[0], dag[-1]

    # ------------------------------------------------------------------ title
    story.append(Spacer(1, 6))
    P('NeuroSnap: Vision-Guided Learned Trajectory Generation for Autonomous Quadrotor Gate Navigation', title)
    P('Project Implementation Report', sub)
    P('<b>Name:</b> Aryan Amit Tiwari', centre)
    P('<b>Registration No.:</b> 23BAI1409', centre)
    P('<b>Course:</b> BCSE428L – Autonomous Drones', centre)
    story.append(Spacer(1, 10))

    # ------------------------------------------------------------------ 1 links
    H1('1. Project Links')
    gh = link(github) if github else link('https://github.com/AryanAT13/Neurosnap')
    rows = [('Source code (GitHub)', gh),
            ('Repository README', 'README.md (results, figures, how to run)'),
            ('Demo video', 'docs/media/demo.mp4 and docs/media/demo.gif'),
            ('Raw results (CSV / JSON)', 'results/ (benchmark.csv, benchmark_summary.json, planner_eval.json, '
                                         'perception_eval.json, dagger.json, control_analysis.json)'),
            ('Trained models', 'models/ (planner_dagger.pt, gatenet.pt, residual.npz and training logs)'),
            ('Initial report (proposal)', 'Drone_Project_Report_Aryan_Amit_Tiwari_23BAI1409.pdf'),
            ('Reference paper (IEEE)', link('https://doi.org/10.1109/ICRA.2011.5980409'))]
    t = Table([[Paragraph(f'<b>{a}</b>', cell), Paragraph(b, cell)] for a, b in rows],
              colWidths=[PAGE_W * 0.27, PAGE_W * 0.73])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
                           ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#bbbbbb')),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 5),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 5), ('LEFTPADDING', (0, 0), (-1, -1), 9)]))
    story.append(t)
    story.append(Spacer(1, 6))

    # ------------------------------------------------------------------ 2 abstract
    H1('2. Abstract')
    P('This report describes the implementation of <b>NeuroSnap</b> (<i>Neural minimum-Snap</i>), the project '
      'proposed in the initial report. A simulated Crazyflie-class quadrotor must fly through a randomised course '
      'of gates using only a forward-facing camera and an approximate, noisy map of the track. A small '
      'convolutional network (GateNet) finds the four corners of every gate in each frame. PnP with the known '
      'gate size turns the corners into 3-D gate positions, and a Kalman filter fuses them with the prior map. '
      'A Transformer trajectory generator then replaces the reference paper\'s minimum-snap optimisation: in a '
      'single forward pass it predicts segment durations and knot derivatives, which a fixed Hermite decoder '
      'turns into a smooth, C3-continuous trajectory. A cascaded PID controller with a learned drag '
      'feed-forward flies the trajectory on a full 6-DOF model.')
    P(f'Across {n_missions} simulated missions on {nb} unseen courses, the full stack completed '
      f'<b>{pct(ns["success"])}</b> of courses, against {pct(qp["success"])} for the classical minimum-snap '
      f'planner with the same perception and 0 % when flying the prior map alone. It tracked its reference '
      f'with {100 * ns["track_rmse"]:.1f} cm RMSE (classical: {100 * qp["track_rmse"]:.1f} cm). Offline, the '
      f'learned time allocation is within {100 * (hyb["duration_ratio_median"] - 1):.1f} % of the optimiser, '
      f'and in hybrid mode the trajectories match it ({hyb["snap_ratio_median"]:.2f}× optimal snap) at '
      f'{teacher["ms_mean"] / hyb["ms_mean"]:.1f}× lower compute. Two additions made the learned planner '
      'work in closed loop: <b>DAgger</b> against receding-horizon drift, and an analytic <b>plan verifier</b>. '
      f'The detector reaches {perc["precision"]:.2f} precision and {perc["recall"]:.2f} recall, and one camera '
      f'frame is processed in {perc["latency_ms_mean"]:.1f} ms on a laptop CPU.')

    # ------------------------------------------------------------------ 3 recap
    H1('3. Reference Paper and Objective (recap)')
    P('The project builds on D. Mellinger and V. Kumar, "<i>Minimum Snap Trajectory Generation and Control for '
      'Quadrotors</i>", IEEE ICRA 2011, pp. 2520–2525 (Best Conference Paper). The paper uses differential '
      'flatness to plan in the space of position and yaw. It represents the trajectory as piecewise '
      'polynomials, minimises the integral of squared snap as a quadratic program, optimises the segment '
      'durations, and tracks the result with a nonlinear controller. The method is powerful but mathematically '
      'heavy, assumes the waypoints are given, and must be re-solved whenever they change.')
    P('The objective set in the initial report was a software pipeline with these parts:')
    bullets(['detect gates from a single onboard camera with a deep network and estimate their pose;',
             'convert detections into 3-D waypoints and fuse them with a prior map;',
             'generate a smooth, dynamically feasible trajectory with a neural network trained by imitation of '
             'the minimum-snap solver, replacing the run-time QP and time allocation;',
             'track it with a PID controller, optionally with a learned residual model, behind a safety layer;',
             'evaluate detection quality, trajectory quality, tracking error, success rate and compute time against '
             'the classical minimum-snap planner and a straight-line planner.'], numbered=True)
    P('All five parts are implemented and evaluated below. The learned residual model, listed as a stretch goal, '
      'is also complete.')
    note('<b>Deviations from the proposal.</b> (1) The proposal named gym-pybullet-drones. The project instead uses '
         'its own 6-DOF simulator with the same Crazyflie CF2X parameters, because a self-written renderer gives '
         'exact corner labels for every frame and the PyBullet build was not needed for anything else. (2) The '
         'proposal named a YOLO-family detector with a pose-regression head. GateNet is a CenterNet-style '
         'single-stage detector that regresses the four opening corners directly; PnP then gives the exact pose '
         'from geometry instead of a learned regressor. (3) DAgger and the plan verifier were not in the proposal; '
         'they were added after the first closed-loop results (Section 10).')

    # ------------------------------------------------------------------ 4 overview
    H1('4. System Overview')
    figure('architecture.png', 'Figure 1. NeuroSnap architecture. Top two rows: the online loop (camera and '
           'perception at 15 Hz, re-planning at 10 Hz, control at 500 Hz, physics at 1 kHz). Bottom row: '
           'offline training.')
    P('The system is written in Python (NumPy, SciPy, PyTorch, OpenCV, Matplotlib) and organised as independent '
      'modules:')
    bullets(['a simulated quadrotor and FPV camera producing images;',
             'a perception module producing gate detections and PnP poses;',
             'a gate map (Kalman filter) producing gate-position estimates;',
             'a planning module producing time-parameterised reference trajectories, checked by the verifier;',
             'a cascaded flight controller producing motor commands.'])
    P('Every component was built from scratch for this project: the simulator and renderer, the minimum-snap '
      'teacher, the planner dataset, both neural networks and the residual model, DAgger, the mission runner, the '
      'benchmark and the dashboard. The code is about 3,500 lines with 14 unit and integration tests.')
    figure('dashboard_mid.png', 'Figure 2. The live dashboard 6.3 s into a mission. Left: top view with the true '
           'gates (orange), the prior map (x), the current gate estimates with 1-sigma circles (cyan), the current '
           'plan (green) and the flown path (white). Right: the onboard camera with GateNet detections and PnP '
           'ranges, and the flight HUD.', width=PAGE_W * 0.92)

    # ------------------------------------------------------------------ 5 implementation
    H1('5. Implementation')
    H2('5.1 Quadrotor model (Syllabus Modules 1 and 4)')
    P('The vehicle is a rigid-body Newton–Euler model of the Bitcraze Crazyflie 2.x in X configuration, with '
      'the parameters of the gym-pybullet-drones CF2X model. Each rotor produces thrust T<sub>i</sub> = '
      'k<sub>f</sub> n<sub>i</sub><super>2</super> and drag torque Q<sub>i</sub> = k<sub>m</sub> '
      'n<sub>i</sub><super>2</super> (n in RPM). Rotor speed follows the command through a first-order lag, and '
      'rotor drag acts against the body-frame velocity in proportion to the summed rotor speed:')
    P('m dv/dt = R [0, 0, ΣT<sub>i</sub>]<super>T</super> − m g e<sub>z</sub> + F<sub>drag</sub> + '
      'F<sub>ext</sub>,&nbsp;&nbsp;&nbsp; J dω/dt = τ − ω × Jω,&nbsp;&nbsp;&nbsp; '
      'dn<sub>i</sub>/dt = (n<sub>i,cmd</sub> − n<sub>i</sub>) / τ<sub>m</sub>', eq)
    P('The state is integrated with semi-implicit Euler at 1 kHz, with quaternion attitude. Each mission draws a '
      'random constant wind (σ = 0.3 m/s), and the controller sees position and velocity with small '
      'Gaussian noise.')
    table([('Parameter', 'Value', 'Parameter', 'Value'),
           ('Mass', '27 g', 'Airframe / arm length', 'Quad-X / 39.7 mm'),
           ('Inertia (Ixx, Iyy, Izz)', '1.4, 1.4, 2.17 × 10<super>-5</super> kg·m²', 'k<sub>f</sub> / k<sub>m</sub>',
            '3.16×10<super>-10</super> / 7.94×10<super>-12</super>'),
           ('Thrust-to-weight', '2.25', 'Motor time constant', '20 ms'),
           ('Hover / max rotor speed', '14,476 / 21,714 RPM', 'Rotor drag (xy, z)', '9.18 / 10.31 ×10<super>-7</super>'),
           ('Camera', '192×144 px, 90° HFOV, tilted 15° up, 15 Hz', 'Physics / control rate', '1 kHz / 500 Hz'),
           ('Gates', '0.8 m opening, 0.15 m frame, on a stand', 'Limits (planner)', 'v ≤ 3 m/s, a ≤ 6 m/s²')],
          [0.22, 0.30, 0.22, 0.26])

    H2('5.2 Flight controller (Syllabus Modules 2 and 3)')
    P('The reference paper tracks its trajectories with a hand-designed nonlinear controller. NeuroSnap uses a '
      'standard PX4/DSL-style cascade instead:')
    bullets(['position and velocity PID with acceleration feed-forward from the trajectory, giving a desired thrust '
             'vector (tilt limited to 45°);',
             'desired attitude from the thrust direction and the yaw reference (heading towards the next gate);',
             'attitude P on the SO(3) error e<sub>R</sub> = ½ vee(R<sub>d</sub><super>T</super>R − '
             'R<super>T</super>R<sub>d</sub>), giving body-rate set-points (≤ 12 rad/s);',
             'body-rate PI and a mixer that inverts the 4×4 thrust/torque allocation matrix.'])
    P('The position integrator uses conditional integration: only errors under 25 cm are integrated. An early '
      'version wound up on large steps and settled slowly (Section 10). The residual model of Section 5.9 is '
      'subtracted from the commanded acceleration as a feed-forward term.')

    H2('5.3 FPV camera and self-labelled training data (Syllabus Module 7)')
    P('The onboard camera is a small renderer written in NumPy and OpenCV. The ground and sky are ray-cast per '
      'pixel onto the plane z = 0 with a procedural texture (noise, checkerboard or tiles), distance fog and a sky '
      'gradient. Gates, stands and distractor boxes and pillars are drawn as flat-shaded polygons, with back-face '
      'culling, near-plane clipping (Sutherland–Hodgman) and painter\'s-algorithm ordering. Every scene is '
      'domain-randomised: gate colour, sky, ground texture, light direction, fog, brightness, contrast, colour '
      'gain, sensor noise and motion blur.')
    P('Alongside the image, the renderer outputs exact labels: the projected opening corners (ordered as seen, so '
      'a gate seen from behind keeps a consistent corner order) and the gate pose in the camera frame. No manual '
      'annotation was needed. Training frames are generated on the fly in four DataLoader workers from camera '
      'poses sampled as they occur in flight (between consecutive gates, looking roughly at the next one, with '
      'manoeuvring roll and pitch), so GateNet never sees the same image twice.')
    figure('fpv_samples.png', 'Figure 3. Onboard views rendered with the demo style. The far gates are seen '
           'through the openings of the near ones.')

    H2('5.4 Gate perception: GateNet and PnP')
    table([('Network', 'Design', 'Parameters', 'CPU time'),
           ('GateNet', 'CenterNet-style, anchor-free: residual encoder (5 stages, strides 2–32) with a '
                       'feature-pyramid decoder to stride 4. Heads: centre heat-map (focal loss), sub-cell offset '
                       'and 8 corner offsets (L1). Trained on the laptop GPU (Apple MPS), 25 epochs × 250 '
                       'steps × 32 images = 200k fresh renders, about 35 min.', '2.92 M',
            f'{perc["latency_ms_mean"]:.1f} ms*')],
          [0.14, 0.60, 0.13, 0.13], right_from=2)
    P('* Including heat-map decoding and PnP for all detections. Peaks are extracted with 3×3 max-pool '
      'non-maximum suppression. For each detection the four corners and the known 0.8 m opening go into '
      '<font name="Mono">cv2.solvePnP</font> with the IPPE-square solver, which gives the gate centre and normal '
      'in the camera frame. Keeping the geometry in PnP means the network only has to learn image keypoints, '
      'which transfers far better than regressing 3-D pose directly.')

    H2('5.5 Gate map (Kalman filter)')
    P('Before take-off the drone receives a prior map: every true gate centre plus Gaussian noise (σ = 0.45 m '
      'horizontally, 0.2 m vertically). This is the only information the "blind" ablation has. Each PnP fix within '
      '7 m is transformed into the world frame with the estimated camera pose. It is associated with the nearest '
      'not-yet-passed gate by Mahalanobis distance (gate χ² = 16) and fused by a per-gate Kalman update. '
      'The measurement covariance grows with range, σ(d) = 0.04 + 0.012 d² m, which follows the PnP '
      'depth error measured in Section 7.3.')

    H2('5.6 Minimum-snap teacher (the reference paper, used offline)')
    P('The paper\'s method is implemented in its unconstrained knot-derivative form. Each segment is a '
      '7th-order polynomial fixed by position, velocity, acceleration and jerk at its two ends. On normalised time '
      'τ = t/T the boundary values become y<sub>d</sub> = x<sub>d</sub> T<super>d</super>, the coefficients '
      'are a = A y with a constant 8×8 matrix A, and the integrated squared snap is a closed-form quadratic:')
    P('J = ∫ (d<super>4</super>p / dt<super>4</super>)<super>2</super> dt = T<super>−7</super> y<super>T</super> H y,&nbsp;&nbsp; '
      'H = A<super>T</super> Q<sub>1</sub> A,&nbsp;&nbsp;&nbsp; G<sub>ij</sub>(T) = H<sub>ij</sub> '
      'T<super>d<sub>i</sub>+d<sub>j</sub>−7</super>', eq)
    P('With the waypoints and the current velocity and acceleration fixed, the free derivatives (initial jerk and '
      'v, a, j at every interior waypoint) follow from one linear solve shared by x, y and z. The segment '
      'durations are then optimised on log T with L-BFGS for F(T) = J*(T) + ρ ΣT (ρ = 40). The '
      'gradient comes for free from the envelope theorem: dJ*/dT<sub>s</sub> = x<sub>s</sub><super>T</super> '
      '(dG<sub>s</sub>/dT) x<sub>s</sub>. A feasibility step stretches the durations and re-solves until '
      '|v| ≤ 3 m/s and |a| ≤ 6 m/s². The teacher labelled 80,000 training and 4,000 test '
      'problems in 157 s on six cores. Half of the problems start mid-flight from a teacher trajectory with '
      'perturbed velocity and acceleration, so re-planning states are covered.')

    H2('5.7 Neural trajectory generator')
    P('The waypoints are expressed in a canonical frame: origin at the current position, x-axis towards the first '
      'waypoint. Each waypoint becomes a 16-dimensional token (position, offset to the previous waypoint, '
      'segment length, start and end flags, the initial velocity and acceleration on the first token, and its '
      'index). A 4-layer pre-norm Transformer encoder (d = 128, 4 heads, 568k parameters) maps the tokens to ten '
      'outputs per waypoint: log T of the incoming segment and the 3 × (v, a, j) knot derivatives.')
    P('The outputs pass through a fixed, differentiable <b>Hermite decoder</b>: the same 7th-order '
      'parametrisation the teacher uses. The decoder overwrites the positions with the waypoints, the first '
      'velocity and acceleration with the current state, and the final derivatives with zero. So every network '
      'output passes exactly through the waypoints, starts from the vehicle\'s state and is C3-continuous by '
      'construction. Derivatives are predicted in duration-normalised, standardised units (x<sub>d</sub> '
      'τ<super>d</super>, with τ a geometric time scale from segment lengths). With fixed scales the jerk '
      'targets had a standard deviation of about 0.02, so jerk errors barely counted in the loss (Section 10).')
    P('Two inference modes are evaluated: <b>full neural</b> (durations and derivatives from the network, no solver '
      'on board) and <b>hybrid</b> (durations from the network, derivatives from one closed-form QP solve). Both '
      'pass through the same safety layer as the classical planner.')

    H2('5.8 DAgger for receding-horizon re-planning')
    P('The network is trained on states that lie on teacher trajectories. In flight, however, it re-plans every '
      '0.1 s from states on its <i>own</i> previous plan. Small duration biases compound, and each re-plan comes '
      'out slightly slower than the last. This is the classic covariate shift of imitation learning. NeuroSnap '
      'applies DAgger (Ross et al., 2011) in three rounds. Each round runs the current network in planner-only '
      'receding-horizon rollouts on 1,500 fresh courses (about 240,000 visited states). It labels 30,000 of those '
      'states with the teacher, adds them to the training set and fine-tunes for 15 epochs.')

    H2('5.9 Safety layer, plan verifier and learned residual dynamics')
    P('<b>Safety layer.</b> If a trajectory exceeds the velocity or acceleration limit, its durations are '
      'stretched and, for the QP planners, the derivatives are re-solved. The start state is kept exactly.')
    P('<b>Plan verifier.</b> Before any re-plan, classical or learned, replaces the current trajectory, it must '
      'pass two cheap analytic checks. It must not reverse direction along its first segment (the '
      'overshoot-and-return loop), and it must not reach the next gate more than 1.4× later than the current '
      'plan. Otherwise the current plan is kept for another 0.1 s. Re-planning also stops within 0.7 m of a gate.')
    P(f'<b>Residual dynamics.</b> A 4→32→32→3 tanh MLP maps body-frame velocity and summed rotor '
      f'speed to the acceleration that the thrust model does not explain (rotor drag and wind). It was trained on '
      f'{res["samples"]:,} samples from 35 logged flights and explains '
      f'{100 * (1 - res["model_rmse"] / res["residual_rms"]):.0f} % of the residual '
      f'({res["residual_rms"]:.3f} → {res["model_rmse"]:.3f} m/s² RMS on held-out flights). It runs '
      f'in NumPy at the 500 Hz control rate.')

    H2('5.10 Live demo dashboard')
    P('The dashboard runs the complete loop and can be shown live in an OpenCV window or recorded to MP4 or GIF. '
      'It shows the top view (true gates, prior map, gate estimates with uncertainty, the current plan and the '
      'flown path), the onboard camera with detections, PnP ranges and confidence, and a HUD with time, speed, '
      'altitude, gates passed, planner and perception latency and a speed trace. Command-line options switch the '
      'planner (neural, hybrid, QP with optimised or heuristic times, straight line), the perception mode '
      '(camera, blind, oracle) and the residual model.')
    figure_pair('dashboard_early.png', 'Figure 4a. Early in the mission: gate estimates (cyan) are being pulled '
                'from the prior (x) towards the true gates.',
                'dashboard_end.png', 'Figure 4b. Mission complete: all five gates passed, 13.9 s; the speed trace '
                'shows the smooth profile of the learned trajectory.')

    # ------------------------------------------------------------------ 6 setup
    H1('6. Experimental Setup and Metrics')
    P(f'Courses are generated at random with 3–7 gates, 2.5–5 m apart, with heading changes up to '
      f'±60°, heights from 0.8 to 2.0 m, gate yaw jitter of ±20° and 4–12 distractor '
      f'objects. The finish point lies 2 m after the last gate. The benchmark uses {nb} course seeds that were '
      f'never used during development. Every method flies exactly the same courses, prior maps and wind. Eleven '
      f'methods were compared:')
    bullets(['straight-line waypoint follower at 2.2 m/s and at 3.0 m/s (the baseline from the proposal);',
             'minimum-snap QP with heuristic durations, and with optimised durations (the reference paper);',
             'the neural planner in hybrid and full mode, and the full mode without DAgger;',
             '<b>NeuroSnap</b>: full neural planner + GateNet + residual model;',
             'NeuroSnap without the plan verifier, NeuroSnap flying the prior map only ("blind"), and NeuroSnap with '
             'the true gate positions ("oracle", an upper bound).'])
    P('A mission succeeds if the drone passes every gate in order, without touching a frame, a stand or the '
      'ground, and stops within 0.35 m of the finish within 40 s. A crossing of a gate plane outside the frame '
      'counts as a missed gate. The metrics are those listed in Section 8.7 of the initial report:')
    bullets(['detection precision and recall, corner error and PnP position error against range;',
             'trajectory quality: snap cost and duration relative to the minimum-snap optimum;',
             'tracking performance: RMSE of the flown path against the reference;',
             'mission success rate, gates passed and lap time;',
             'compute time: planner time per call against the QP, and perception latency;',
             'ablations: Transformer against MLP, snap loss, DAgger, verifier, residual model, perception mode.'])

    # ------------------------------------------------------------------ 7 results
    H1('7. Results')
    H2(f'7.1 Learned planner against the optimiser ({plan["n_problems"]:,} held-out problems)')
    order = [('Min-snap QP, heuristic T', 'Min-snap QP, heuristic durations'),
             ('Min-snap QP, optimised T (teacher)', 'Min-snap QP, optimised durations (teacher)'),
             ('Transformer, no snap loss, DAgger (full neural)', '<b>Transformer + DAgger, full neural</b>'),
             ('Transformer, no snap loss, DAgger + QP derivatives (hybrid)', '<b>Transformer + DAgger, hybrid</b>'),
             ('Transformer, no snap loss (full neural)', 'Transformer, behaviour cloning only'),
             ('Transformer, snap loss 0.1 (full neural)', 'Transformer, + snap-gap loss 0.1'),
             ('Transformer, snap loss 0.5 (full neural)', 'Transformer, + snap-gap loss 0.5'),
             ('MLP, snap loss 0.5 (full neural)', 'MLP (same decoder, snap loss 0.5)')]
    rows = [('Planner', 'Snap / optimal (median)', 'Snap p90', 'Duration / optimal', 'ms per call (mean / p95)')]
    for key, label in order:
        s = ps[key]
        rows.append((label, f'{s["snap_ratio_median"]:.2f}×', f'{s["snap_ratio_p90"]:.1f}×',
                     f'{s["duration_ratio_median"]:.3f}', f'{s["ms_mean"]:.2f} / {s["ms_p95"]:.2f}'))
    table(rows, [0.38, 0.16, 0.11, 0.15, 0.20], highlight=(3, 4), right_from=1)
    P(f'The <b>hybrid mode matches the optimiser</b> ({hyb["snap_ratio_median"]:.2f}× snap, durations within '
      f'0.1 %) at {teacher["ms_mean"] / hyb["ms_mean"]:.1f}× lower mean and '
      f'{teacher["ms_p95"] / hyb["ms_p95"]:.1f}× lower p95 compute: the network replaces the iterative time '
      f'allocation, and a single linear solve recovers the optimal derivatives. The <b>full-neural mode</b> needs '
      f'no solver at all. Its durations are just as accurate, but its trajectories carry '
      f'{full["snap_ratio_median"]:.1f}× the minimum snap, because snap is very sensitive to small jerk errors '
      f'at the knots. Attention matters: the MLP with the same decoder is 8× worse. The snap-gap loss is a '
      f'negative result. It lowers snap at the teacher\'s durations, but it makes the predicted durations less '
      f'accurate, and since snap scales with T<super>−7</super>, duration accuracy dominates end to end.')
    figure('planner_eval.png', 'Figure 5. Snap cost and duration relative to the optimiser (box plots, outliers '
           'hidden) and compute time per planning call, single CPU thread.')
    H2('7.2 DAgger and receding-horizon drift')
    rows = [('Planner', 'Drift median', 'Drift p90', 'Blow-ups (> 3×) / 100')]
    rows.append(('Teacher (min-snap, optimised durations)', f'{d0["teacher"]["median"]:.2f}', f'{d0["teacher"]["p90"]:.2f}',
                 str(d0['teacher']['blowups'])))
    rows.append(('Transformer, behaviour cloning', f'{d0["drift"]["median"]:.2f}', f'{d0["drift"]["p90"]:.2f}',
                 str(d0['drift']['blowups'])))
    for d in dag[1:]:
        rows.append((f'+ DAgger round {d["iter"]} ({d["states"]:,} states)', f'{d["drift"]["median"]:.2f}',
                     f'{d["drift"]["p90"]:.2f}', str(d['drift']['blowups'])))
    table(rows, [0.46, 0.17, 0.17, 0.20], highlight=(len(rows) - 1,), right_from=1)
    P(f'Drift is the time to finish when re-planning every 0.1 s, divided by the duration of the first plan '
      f'(planner-only rollouts on 100 unseen courses). DAgger cuts the slow-down from '
      f'{d0["drift"]["median"]:.2f}× to {d3["drift"]["median"]:.2f}× and the blow-ups from '
      f'{d0["drift"]["blowups"]} to {d3["drift"]["blowups"]}. It also improved the offline snap ratio from 2.31× '
      f'to {full["snap_ratio_median"]:.2f}×. The teacher itself drifts by 1.25×, because its '
      f'time-optimal tail problems are re-solved from a fixed start state.')

    H2(f'7.3 Perception ({perc["frames"]:,} held-out frames, {perc["gates"]:,} gates within 8 m)')
    rows = [('Metric', 'Value')]
    rows += [('Detection precision / recall', f'{perc["precision"]:.2f} / {perc["recall"]:.2f}'),
             ('Corner keypoint error (median)', f'{perc["corner_px_median"]:.2f} px'),
             ('Gate-centre error after PnP (median, all ranges)', f'{100 * perc["pos_err_median"]:.1f} cm'),
             ('Latency: GateNet + decoding + PnP (CPU, mean / p95)',
              f'{perc["latency_ms_mean"]:.1f} / {perc["latency_ms_p95"]:.1f} ms')]
    table(rows, [0.62, 0.38], right_from=1)
    rows = [('Range to gate', 'Frames', 'Corner error', 'Position error (median)', 'Position error (p90)', 'KF σ used')]
    for b in perc['by_range']:
        rows.append((b['range'], str(b['n']), f'{b["corner_px"]:.2f} px', f'{100 * b["pos_err_median"]:.1f} cm',
                     f'{100 * b["pos_err_p90"]:.1f} cm', f'{100 * b["model_sigma"]:.1f} cm'))
    table(rows, [0.17, 0.11, 0.15, 0.21, 0.20, 0.16], right_from=1)
    P('Corner error in pixels is almost constant with range, so the 3-D error grows roughly with distance '
      'squared, as PnP theory predicts. This is why far fixes are down-weighted in the Kalman filter and fixes '
      'beyond 7 m are ignored. Close gates have a larger pixel error but still the best 3-D accuracy (4.8 cm), '
      'even when their corners fall outside the image.')
    figure('perception_examples.png', 'Figure 6. Held-out validation frames with GateNet detections (green) and '
           'PnP range, across randomised colours, textures, lighting, noise and blur.')
    figure('perception_eval.png', 'Figure 7. PnP gate-centre error against range with the Kalman-filter '
           'measurement model (left) and the corner-keypoint error distribution (right).', width=PAGE_W * 0.9)

    H2(f'7.4 Closed-loop benchmark ({n_missions} missions)')
    names = [('line+cnn', 'Straight line 2.2 m/s'), ('line_fast+cnn', 'Straight line 3.0 m/s'),
             ('qp_heur+cnn', 'Min-snap, heuristic T'), ('qp_opt+cnn', 'Min-snap, optimised T (paper)'),
             ('nn_hybrid+cnn', 'Neural hybrid'), ('nn_full+cnn', 'Neural full'),
             ('nn_full_bc+cnn', 'Neural full, no DAgger'), ('neurosnap', '<b>NeuroSnap</b> (+ residual)'),
             ('neurosnap_noverify', 'NeuroSnap, no verifier'), ('neurosnap_blind', 'NeuroSnap, blind'),
             ('neurosnap_oracle', 'NeuroSnap, oracle gates')]
    rows = [('Method', 'Success', 'Crash', 'Missed', 'Gates', 'Lap (s)', 'RMSE (cm)', 'Gate err (cm)')]
    for key, label in names:
        s = bench[key]
        rows.append((label, pct(s['success']), pct(s['crash']), pct(s['missed']), pct(s['gates_frac']),
                     f'{s["lap_time"]:.1f}' if s['lap_time'] else '–',
                     f'{100 * s["track_rmse"]:.1f}' + ('*' if s['track_rmse'] > 1 else ''),
                     f'{100 * s["gate_est_err"]:.1f}' if s['gate_est_err'] is not None else '–'))
    table(rows, [0.27, 0.10, 0.09, 0.09, 0.09, 0.09, 0.12, 0.15], highlight=(8,), right_from=1)
    P('All methods except "blind" and "oracle" use GateNet perception. * One run without the verifier accepted a '
      'bad plan and diverged, which dominates the mean.')
    figure('benchmark.png', 'Figure 8. Success rate, gates passed, tracking RMSE (log scale) and planner compute '
           'time per method. Planner times here include seven missions running in parallel.')
    P(f'<b>Perception is what makes the task solvable.</b> Flying the prior map completes 0 % of courses. With '
      f'GateNet, PnP and the Kalman map, the gate-position error falls from {100 * bench["neurosnap_blind"]["gate_est_err"]:.0f} '
      f'cm to {100 * ns["gate_est_err"]:.1f} cm and NeuroSnap completes {pct(ns["success"])}. <b>The learned stack '
      f'comes within {100 * (qp["success"] - ns["success"]):.1f} points of the classical planner</b>, with '
      f'{qp["track_rmse"] / ns["track_rmse"]:.1f}× lower tracking error (thanks to the residual model) and no '
      f'optimiser on board. <b>DAgger and the verifier are both needed:</b> removing DAgger costs '
      f'{100 * (bench["nn_full+cnn"]["success"] - bench["nn_full_bc+cnn"]["success"]):.1f} points, and removing the '
      f'verifier costs {100 * (ns["success"] - bench["neurosnap_noverify"]["success"]):.0f}. NeuroSnap is still '
      f'about {100 * (ns["lap_time"] / qp["lap_time"] - 1):.0f} % slower per lap than the classical planner '
      f'({ns["lap_time"]:.1f} s against {qp["lap_time"]:.1f} s). This is the remaining receding-horizon drift: '
      f'without re-planning (oracle) the same network flies the course in {bench["neurosnap_oracle"]["lap_time"]:.1f} s.')
    note('<b>The straight-line follower is a strong baseline in this setup.</b> The 0.8 m gates are generous for a '
         '12 cm vehicle, and a straight line goes exactly through the gate centres, so it never fails and is the '
         'fastest. The price is 14.5–24.7 cm of tracking error, 4–15× more than the smooth '
         'planners: the vehicle cannot follow corners it was never planned to fly. This dynamic infeasibility is '
         'exactly what minimum-snap planning exists to remove, and with tighter gates or faster flight it would '
         'cause failures. The result is reported as measured.')
    figure('benchmark_trajectories.png', 'Figure 9. The paths flown by all eleven methods on the same course (black: '
           'true gates, x: prior map). Only the blind variant, which aims at the prior map, fails.')

    H2('7.5 Real-time performance')
    P(f'Per camera frame, GateNet with decoding and PnP takes {perc["latency_ms_mean"]:.1f} ms on the CPU (p95 '
      f'{perc["latency_ms_p95"]:.1f} ms). The neural planner takes {full["ms_mean"]:.2f} ms per re-plan (hybrid '
      f'{hyb["ms_mean"]:.2f} ms; classical optimiser {teacher["ms_mean"]:.2f} ms, p95 {teacher["ms_p95"]:.2f} ms), '
      f'and the residual model runs in NumPy at 500 Hz. The complete perception-to-plan chain therefore takes about '
      f'10 ms, well inside the 66 ms budget of a 15 Hz camera.')

    # ------------------------------------------------------------------ 8 control
    H1('8. Stability and Control Analysis (Syllabus Module 5)')
    P('<b>Step responses.</b> The cascade was tested on the full nonlinear model with attitude-level steps '
      '(the position loop bypassed), an altitude step through the full cascade, and a horizontal gust at hover.')
    rr, pr, yr, ar, gr = ctrl['roll'], ctrl['pitch'], ctrl['yaw'], ctrl['altitude'], ctrl['gust']
    rl, al, ag = ctrl['roll_rate_loop'], ctrl['roll_attitude_loop'], ctrl['attitude_gain']
    table([('Axis / test', 'Result'),
           ('Roll (lateral) and pitch (longitudinal) 10° step',
            f'rise time {rr["rise_s"]:.2f} s, overshoot {rr["overshoot_pct"]:.1f} %, 5 % settling {rr["settle_5pct_s"]:.2f} s'),
           ('Yaw (directional) 45° step',
            f'rise time {yr["rise_s"]:.2f} s, overshoot {yr["overshoot_pct"]:.1f} %, settling {yr["settle_5pct_s"]:.2f} s'),
           ('Altitude +0.5 m step', f'rise time {ar["rise_s"]:.2f} s, overshoot {ar["overshoot_pct"]:.1f} %, '
                                    f'settling {ar["settle_5pct_s"]:.2f} s'),
           ('Gust: 0.05 N (19 % of weight) for 0.5 s',
            f'peak deviation {100 * gr["peak_dev_m"]:.1f} cm, back within 2 cm after {gr["recover_2cm_s"]:.1f} s'),
           ('Roll rate loop (PI + motor lag + 1-sample delay)',
            f'phase margin {rl["phase_margin_deg"]:.0f}°, gain margin {rl["gain_margin_db"]:.1f} dB, crossover '
            f'{rl["crossover_rad_s"]:.0f} rad/s'),
           ('Roll attitude loop', f'phase margin {al["phase_margin_deg"]:.0f}°, gain margin '
                                  f'{al["gain_margin_db"]:.1f} dB, crossover {al["crossover_rad_s"]:.0f} rad/s'),
           ('Stability boundary', f'attitude loop unstable only for K<sub>att</sub> > {ag["unstable_above"]:.0f} '
                                  f'(design {ag["design"]:.0f}: {ag["unstable_above"] / ag["design"]:.1f}× margin, '
                                  f'consistent with the {al["gain_margin_db"]:.1f} dB gain margin)')],
          [0.42, 0.58])
    figure('control_steps.png', 'Figure 10. Closed-loop responses on the nonlinear model: roll, pitch, yaw, '
           'altitude and gust rejection.')
    P('<b>Linear analysis.</b> About hover the roll axis reduces to a double integrator, because the controller '
      'cancels the inertia. The rate loop is PI control through the motor lag (τ = 20 ms) and one 2 ms '
      'control period of delay (first-order Padé), and the attitude loop is P control around the closed rate '
      'loop. The bare airframe has no restoring moment, so all stability comes from feedback. The rate loop has '
      'the thinner margin (37°), which is the main limit on more aggressive attitude gains.')
    figure('control_bode.png', 'Figure 11. Roll-axis Bode plots of the rate and attitude loops (left, centre) and '
           'closed-loop poles as the attitude gain varies from 0.5 to 200 (right).')

    # ------------------------------------------------------------------ 9 syllabus
    H1('9. Mapping to the BCSE428L Syllabus')
    table([('Module', 'Where it appears in NeuroSnap'),
           ('1 – Introduction (types, airframe, batteries, motors, ESC, propellers)',
            'Quad-X Crazyflie airframe, rotor thrust/torque model, motor lag, rotor drag (Sec. 5.1)'),
           ('2 – Design fundamentals (flight controllers, RC, FPV, telemetry)',
            'Cascaded flight controller and mixer, FPV camera, dashboard with flight HUD (Sec. 5.2, 5.3, 5.10)'),
           ('3 – Drone basics (flight modes, roll/pitch/yaw, GPS autopilot, intelligent modes)',
            'Autonomous gate-racing mission, attitude and yaw control, waypoint autopilot with re-planning '
            '(Sec. 5.2, 5.8, 5.9)'),
           ('4 – Modelling and control', '6-DOF Newton–Euler model, differential flatness and minimum-snap '
                                              'trajectories, learned residual dynamics (Sec. 5.1, 5.6, 5.9)'),
           ('5 – Stability and control', 'Roll, pitch, yaw and altitude steps, gust rejection, Bode margins, '
                                              'pole locus (Sec. 8)'),
           ('6 – Applications', 'Autonomous drone racing and gate inspection with an approximate map (Sec. 6)'),
           ('7 – Expanding drone abilities (camera, FPV, sensors, speed)',
            'Monocular camera with deep perception, faster planning on board, speed and acceleration limits '
            '(Sec. 5.3–5.7, 7.5)')],
          [0.38, 0.62])

    # ------------------------------------------------------------------ 10 challenges
    H1('10. Challenges Faced and How They Were Solved')
    bullets([
        '<b>Dataset generation hung.</b> Some mid-flight problems with perturbed start states produced degenerate '
        'teacher solutions, whose limit stretching exploded the trajectory duration. Sampling is now capped, and '
        'problems that need more than 3× stretching are rejected.',
        '<b>Jerk errors were invisible to the loss.</b> With fixed output scales, the normalised jerk targets had a '
        'standard deviation of 0.02, and the first planner reached 9× the optimal snap. Predicting '
        'duration-normalised, standardised derivatives fixed the scaling. A first attempt that normalised with the '
        '<i>predicted</i> durations coupled the two outputs and made timing worse; a geometric time scale from the '
        'inputs solved it.',
        '<b>The snap-gap loss did not help.</b> A fair comparison at 60 epochs showed that plain imitation gives '
        'the best end-to-end snap, because duration accuracy dominates. The loss was kept as an ablation and '
        'reported as a negative result.',
        '<b>The drone kept slowing down in flight.</b> Receding-horizon re-planning compounded small duration '
        'biases (1.90× drift). DAgger brought this down to 1.44×.',
        '<b>Overshoot-and-return loops.</b> Just before a gate, the network sometimes predicted a long first segment '
        'while the start velocity was fixed. The only polynomial that fits overshoots and comes back, and the '
        'drone hit the gate it had just passed. The analytic plan verifier rejects such plans.',
        '<b>The first verifier was unfair to the baselines.</b> A rule that rejected any stretched plan blocked 34 % '
        'of the classical planner\'s re-plans and dropped its success to 77.5 %. The rule was removed, since the '
        'reversal check already catches the loop, and the whole benchmark was re-run.',
        '<b>Integrator wind-up.</b> On large steps the position integrator wound up and the drone settled slowly. '
        'Conditional integration and a lower integral gain fixed it.',
        '<b>Non-deterministic benchmark.</b> Renderer noise used the global NumPy RNG, so re-running a mission gave '
        'different outcomes. Every renderer now owns a seeded generator, and missions are reproducible.',
        '<b>Unfair perception scoring.</b> Correct detections of gates beyond 8 m were counted as false positives. '
        'They are now ignored, as in the COCO "ignore" convention, which raised measured precision from 0.74 to 0.89.'])

    # ------------------------------------------------------------------ 11 limitations
    H1('11. Limitations and Future Work')
    bullets([
        '<b>Simulation only.</b> The renderer is far simpler than a real camera (no lens distortion, rolling shutter '
        'or real textures). Real flight would need GateNet fine-tuned on real images and a real state estimator '
        'instead of the simulator\'s state plus noise.',
        '<b>Generous gates.</b> The 0.8 m opening makes the task forgiving, which is why the straight-line baseline '
        'succeeds. Smaller gates and higher limits would separate the planners more clearly.',
        '<b>Remaining drift.</b> After DAgger, re-planning is still slower than the teacher (1.44× vs 1.25×). '
        'More DAgger rounds, or a loss that rewards consistency between consecutive plans, are the next steps.',
        '<b>The verifier is a heuristic.</b> It catches overshoot loops and sudden slow-downs, but not every bad plan '
        '(one 2.5 % crash remains).',
        '<b>No obstacle avoidance and no yaw planning.</b> Distractor objects are only visual, and the heading simply '
        'points at the next gate. The reference paper also plans yaw. Moving to gym-pybullet-drones or PX4/Gazebo '
        'and real hardware are further future work.'])

    # ------------------------------------------------------------------ 12 how to run
    H1('12. How to Run')
    codeblock('python3.12 -m venv .venv && source .venv/bin/activate\n'
              'pip install -r requirements.txt\n'
              'python scripts/demo.py --seed 3 --live          # live dashboard (trained weights included)\n'
              'python scripts/demo.py --planner qp_opt         # same course with the classical planner\n'
              'python scripts/evaluate_planner.py --n 1000     # offline planner comparison\n'
              'python scripts/evaluate_perception.py --n 1000  # GateNet + PnP evaluation\n'
              'python scripts/benchmark.py --seeds 40          # closed-loop benchmark (11 methods)\n'
              'python scripts/control_analysis.py              # Module 5 analysis\n'
              'python -m pytest -q                             # 14 unit / integration tests\n'
              'sh scripts/reproduce.sh                         # everything from scratch')

    # ------------------------------------------------------------------ 13 conclusion
    H1('13. Conclusion')
    P('The project set out to keep the reference paper\'s core idea, smooth and dynamically feasible '
      'minimum-snap trajectories through waypoints as the interface between perception and control, while '
      'replacing its run-time mathematics with a deep-learning pipeline. NeuroSnap implements this end to end:')
    bullets(['a self-labelled, domain-randomised perception pipeline (GateNet + PnP + Kalman map);',
             'a Transformer that predicts minimum-snap trajectories through a structure-preserving decoder, trained '
             'by imitation of the paper\'s optimiser and corrected with DAgger;',
             'an analytic verifier and safety layer around the learned planner;',
             'a cascaded flight controller with a learned drag model, analysed for stability.'])
    P(f'With {pct(ns["success"])} mission success (0 % blind, {pct(qp["success"])} for the classical planner), '
      f'{100 * ns["track_rmse"]:.1f} cm tracking error, near-optimal time allocation at '
      f'{teacher["ms_mean"] / hyb["ms_mean"]:.1f}× lower planning cost and about 10 ms per camera frame, the '
      'pipeline meets the objectives of the initial report. The evaluation also shows honestly where the learned '
      'planner still falls short of the optimiser: lap time and full-neural smoothness.')

    # ------------------------------------------------------------------ references
    story.append(PageBreak())
    story.append(Paragraph('References', h1))
    story.append(rule())
    refs = [
        'D. Mellinger and V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," in <i>Proc. IEEE '
        'ICRA</i>, Shanghai, 2011, pp. 2520–2525. ' + link('https://doi.org/10.1109/ICRA.2011.5980409',
                                                               'doi:10.1109/ICRA.2011.5980409'),
        'C. Richter, A. Bry, and N. Roy, "Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense '
        'Indoor Environments," in <i>Proc. ISRR</i>, 2013.',
        'S. Tankasala and M. Pryor, "Accelerating Trajectory Generation for Quadrotors Using Transformers," '
        '<i>L4DC</i>, 2023. ' + link('https://arxiv.org/abs/2303.15606', 'arXiv:2303.15606'),
        'A. Vaswani et al., "Attention Is All You Need," <i>NeurIPS</i>, 2017. '
        + link('https://arxiv.org/abs/1706.03762', 'arXiv:1706.03762'),
        'S. Ross, G. Gordon, and D. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret '
        'Online Learning," <i>AISTATS</i>, 2011. ' + link('https://arxiv.org/abs/1011.0686', 'arXiv:1011.0686'),
        'X. Zhou, D. Wang, and P. Krähenbühl, "Objects as Points," 2019. '
        + link('https://arxiv.org/abs/1904.07850', 'arXiv:1904.07850'),
        'T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," 2017. '
        + link('https://arxiv.org/abs/1708.02002', 'arXiv:1708.02002'),
        'T. Collins and A. Bartoli, "Infinitesimal Plane-Based Pose Estimation," <i>International Journal of Computer '
        'Vision</i>, vol. 109, 2014.',
        'J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real '
        'World," <i>IROS</i>, 2017. ' + link('https://arxiv.org/abs/1703.06907', 'arXiv:1703.06907'),
        'E. Kaufmann et al., "Beauty and the Beast: Optimal Methods Meet Learning for Drone Racing," <i>ICRA</i>, 2019. '
        + link('https://arxiv.org/abs/1810.06224', 'arXiv:1810.06224'),
        'A. Loquercio et al., "Learning High-Speed Flight in the Wild," <i>Science Robotics</i>, vol. 6, no. 59, 2021. '
        + link('https://doi.org/10.1126/scirobotics.abg5810', 'doi:10.1126/scirobotics.abg5810'),
        'T. Lee, M. Leok, and N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," in '
        '<i>Proc. IEEE CDC</i>, 2010.',
        'J. Panerati et al., "Learning to Fly – a Gym Environment with PyBullet Physics for Reinforcement Learning '
        'of Multi-agent Quadcopter Control," <i>IROS</i>, 2021.',
        'R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," <i>Journal of Basic Engineering</i>, '
        'vol. 82, 1960.',
        'PyTorch, ' + link('https://pytorch.org', 'pytorch.org') + ' · OpenCV, '
        + link('https://opencv.org', 'opencv.org') + ' · SciPy, ' + link('https://scipy.org', 'scipy.org')]
    story.append(ListFlowable([ListItem(Paragraph(r, item), leftIndent=18) for r in refs], bulletType='1',
                              leftIndent=18, bulletFontName='Arial', bulletFontSize=9.5))


def page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Arial', 8)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(2.0 * cm, 1.2 * cm, 'NeuroSnap – Project Implementation Report')
    canvas.drawRightString(A4[0] - 2.0 * cm, 1.2 * cm, f'Aryan Amit Tiwari | 23BAI1409 | Page {doc.page}')
    canvas.restoreState()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--github', default=None, help='public repository URL for the links table')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    build(a.github)
    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.9 * cm, title='NeuroSnap: Project Implementation Report',
                            author='Aryan Amit Tiwari (23BAI1409)')
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print('wrote', a.out)
