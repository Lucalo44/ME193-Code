"""On-screen paddle orientation: fusion filter, motor->scene mapping, the tilt
calibration step, and backhand-aware swing calibration."""

import math

from hardware.orientation import (OrientationFilter, euler_yxz_deg, qaxis, qrotate, scene_mapping,
                                  to_scene)
from hardware.paddle import PaddleBase
from hardware.swing import Calibration, ImuSample, detect_all
from tests.synth import recording
from tools.calibrate_swing import analyze_tilt, build_calibration, permissive

DT = 0.015


def scene_euler(q_motor, up, close_axis):
    return euler_yxz_deg(to_scene(q_motor, scene_mapping(up, close_axis)))


def test_closing_the_face_reads_as_negative_pitch_for_any_mounting():
    # Two different ways of holding the motor: different "up" and face-closing axes.
    for up, close in (((0, 0, 1), (1, 0, 0)), ((0, 1, 0), (0, 0, -1)), ((-1, 0, 0), (0, 1, 0))):
        e = scene_euler(qaxis(close, math.radians(30)), up, close)
        assert abs(e["pitch"] + 30) < 0.5 and abs(e["yaw"]) < 0.5, (up, close, e)
        e = scene_euler(qaxis(up, math.radians(40)), up, close)     # turn about vertical
        assert abs(e["yaw"] - 40) < 0.5 and abs(e["pitch"]) < 0.5, (up, close, e)


def test_filter_integrates_rotation_and_holds_it():
    f = OrientationFilter()
    f.reset((0, 0, 1))
    t = 0.0
    for _ in range(51):
        t += 0.01
        f.update(t, (60, 0, 0), (0, 0, 1.8))        # 0.5 s at 60 deg/s, accelerating (accel ignored)
    angle = math.degrees(2 * math.acos(min(1.0, f.q[0])))
    assert abs(angle - 30) < 1.5
    g = qrotate((f.q[0], -f.q[1], -f.q[2], -f.q[3]), (0, 0, 1))
    for _ in range(200):
        t += 0.01
        f.update(t, (0, 0, 0), g)                   # held still at that tilt
    assert abs(math.degrees(2 * math.acos(min(1.0, f.q[0]))) - angle) < 1.0


def test_accelerometer_cancels_gyro_drift_in_tilt():
    f = OrientationFilter()
    f.reset((0, 0, 1))
    t = 0.0
    for _ in range(500):
        t += 0.01
        f.update(t, (3, 0, 0), (0, 0, 1))           # 3 deg/s bias, paddle actually level
    assert abs(euler_yxz_deg(f.q)["pitch"]) < 4     # would be 15 deg without correction


def test_heading_recenters_when_held_still():
    f = OrientationFilter()
    f.reset((0, 0, 1))
    t = 0.0
    for _ in range(40):
        t += 0.01
        f.update(t, (0, 0, 100), (0, 0, 1))
    turned = abs(math.degrees(f.heading()))
    for _ in range(400):
        t += 0.01
        f.update(t, (0, 0, 0), (0, 0, 1))
    assert abs(math.degrees(f.heading())) < turned / 2


def tilt_recording(raw_per_dps=1.6, angle_deg=45.0, axis="gx"):
    """Rest (gravity +z), slow rotation about a motor axis, then held."""
    out = []
    t = 0.0
    rate = angle_deg / 1.0                            # over 1 s
    for i in range(int(2.5 / DT)):
        if t < 0.5:
            th, w = 0.0, 0.0
        elif t < 1.5:
            th, w = math.radians(rate * (t - 0.5)), rate
        else:
            th, w = math.radians(angle_deg), 0.0
        # Rotating the motor about its x axis by +th: gravity seen by the motor tips the other way.
        ay, az = 1000 * math.sin(th), 1000 * math.cos(th)
        g = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
        g[axis] = w * raw_per_dps
        # The motor's own angles (decidegrees): a rotation about x is roll.
        roll = math.degrees(th) * 10 if axis == "gx" else 0.0
        out.append(ImuSample(t, 0, 0, roll, 0, ay, az, g["gx"], g["gy"], g["gz"]))
        t += DT
    return out


def test_tilt_step_measures_gyro_scale_and_axis():
    samples = tilt_recording(raw_per_dps=1.6)
    rest = [s for s in samples if s.t < 0.5]
    tilt = [s for s in samples if s.t >= 0.4]
    raw_per_dps, axis, angle = analyze_tilt(rest, tilt, Calibration())
    assert abs(angle - 45) < 2
    assert abs(raw_per_dps - 1.6) < 0.1
    assert axis[0] > 0.99


import pytest


@pytest.mark.parametrize("source", ["onboard", "fusion"])
def test_paddle_end_to_end_tilt_shows_closed_face(source, monkeypatch):
    import config as C
    monkeypatch.setattr(C, "ORIENTATION_SOURCE", source)
    cal = Calibration(gyro_raw_per_dps=1.6, close_axis=[1.0, 0.0, 0.0])
    p = PaddleBase(cal)
    samples = tilt_recording(raw_per_dps=1.6)
    for s in [x for x in samples if x.t < 0.5]:
        p._ingest(s)
    p.calibration.zero_from([x for x in samples if x.t < 0.5])
    p._remap()
    for s in [x for x in samples if x.t >= 0.5]:
        p._ingest(s)
    o = p.orientation()
    assert abs(o["pitch"] + 45) < 3, o                # face closed 45 deg
    assert len(o["q"]) == 4


def test_calibration_learns_backhands_separately():
    rest, _ = recording([], rest_s=2.0)
    probe = Calibration()
    probe.zero_from(rest)
    seg = permissive(probe)
    fh = lambda **kw: {"peak_g": 3.5, "peak_dps": 600, **kw}          # forehand: +gz rotation
    bh = lambda **kw: {"peak_g": 3.5, "peak_dps": -600, **kw}         # backhand: -gz rotation
    rec = lambda swings, seed: detect_all(recording(swings, seed=seed)[0], seg)
    sets = {
        "soft": rec([{"peak_g": 2.0, "peak_dps": 400}] * 5, 1),
        "hard": rec([{"peak_g": 6.0, "peak_dps": 900}] * 5, 2),
        "bh_soft": rec([{"peak_g": 2.0, "peak_dps": -400}] * 5, 3),
        "bh_hard": rec([{"peak_g": 6.0, "peak_dps": -900}] * 5, 4),
        # On the backhand the face is flipped: closed reads as the OPPOSITE pitch.
        "closed": rec([fh(pitch_deg=-20 - i) for i in range(5)], 5),
        "open": rec([fh(pitch_deg=20 + i) for i in range(5)], 6),
        "bh_closed": rec([bh(pitch_deg=20 + i) for i in range(5)], 7),
        "bh_open": rec([bh(pitch_deg=-20 - i) for i in range(5)], 8),
        "right": rec([fh(yaw_deg=30 + i) for i in range(5)], 9),
        "left": rec([fh(yaw_deg=-30 - i) for i in range(5)], 10),
    }
    assert all(len(v) == 5 for v in sets.values()), {k: len(v) for k, v in sets.items()}
    cal = build_calibration(rest, sets, Calibration())
    assert cal.stroke_terms and cal.topspin_terms_backhand
    fh_closed = detect_all(recording([fh(pitch_deg=-22)], seed=11)[0], cal)[0]
    bh_closed = detect_all(recording([bh(pitch_deg=22)], seed=12)[0], cal)[0]
    bh_open = detect_all(recording([bh(pitch_deg=-22)], seed=13)[0], cal)[0]
    assert fh_closed.imu_stroke == "forehand" and fh_closed.topspin > 0.5
    assert bh_closed.imu_stroke == "backhand" and bh_closed.topspin > 0.5
    assert bh_open.imu_stroke == "backhand" and bh_open.topspin < -0.5


def test_onboard_angles_are_zyx_euler():
    from hardware.orientation import onboard_quat
    # Pure yaw/pitch/roll rotate about the motor's z / y / x axes.
    for (y, p, r), axis in (((30, 0, 0), (0, 0, 1)), ((0, 30, 0), (0, 1, 0)), ((0, 0, 30), (1, 0, 0))):
        q = onboard_quat(y, p, r)
        assert abs(math.degrees(2 * math.acos(q[0])) - 30) < 1e-6
        assert all(abs(q[i + 1] / math.sin(math.radians(15)) - axis[i]) < 1e-6 for i in range(3))
    # Order: yaw applied first (outermost), then pitch, then roll.
    q = onboard_quat(90, 90, 0)
    x_axis = qrotate(q, (1, 0, 0))
    assert abs(x_axis[2] + 1) < 1e-6        # pitch 90 about the yawed y axis points body x straight down


def test_onboard_orientation_is_relative_to_zero_and_stable():
    from hardware.orientation import OnboardOrientation
    o = OnboardOrientation()
    o.update(-27.3, 5.7, -1.8)              # whatever the motor reports at the ready pose
    o.reset()
    assert o.q == (1.0, 0.0, 0.0, 0.0)
    for _ in range(100):
        o.update(-27.3, 5.7, -1.8)          # held still: no drift, no spinning
    assert abs(o.q[0] - 1.0) < 1e-9


def test_sample_clock_evens_out_bluetooth_bursts():
    from hardware.swing import SampleClock, retime
    # Real pattern: samples every 15 ms, delivered in pairs or batches.
    true_t = [i * 0.015 for i in range(400)]
    arrivals = []
    for i, t in enumerate(true_t):
        group_end = true_t[min(len(true_t) - 1, (i // 2) * 2 + 1)]           # pairs
        if 200 <= i < 210:
            group_end = true_t[209]                                           # one 150 ms batch
        arrivals.append(group_end + 0.004)
    clock = SampleClock()
    stamped = [clock.stamp(a) for a in arrivals]
    gaps = [b - a for a, b in zip(stamped[60:], stamped[61:])]
    assert all(abs(g - 0.015) < 0.0025 for g in gaps), (min(gaps), max(gaps))
    assert all(s <= a for s, a in zip(stamped, arrivals))                    # never in the future
    rt = retime([ImuSample(a, 0, 0, 0, 0, 0, 1000, 0, 0, 0) for a in arrivals])
    assert max(abs((b.t - a.t) - 0.015) for a, b in zip(rt, rt[1:])) < 0.0025
