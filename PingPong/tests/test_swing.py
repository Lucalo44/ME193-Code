"""Swing detector tests on IMU recordings.

Synthetic recordings are written to CSV and read back through the same
load_csv path real recordings use. Real recordings can be dropped into
tests/data/ named like `forehand_soft_5swings.csv` (from tools/imu_logger.py);
each is checked to produce exactly that many swing events, using
swing_calibration.json if present.
"""

import glob
import os
import re

import pytest

from hardware.swing import Calibration, detect_all, load_calibration, load_csv, save_csv
from tests.synth import recording

HERE = os.path.dirname(os.path.abspath(__file__))


def roundtrip(tmp_path, samples):
    path = str(tmp_path / "rec.csv")
    save_csv(path, samples)
    return load_csv(path)


def test_one_event_per_swing(tmp_path):
    samples, centers = recording([{"peak_g": 4, "peak_dps": 600}] * 5)
    events = detect_all(roundtrip(tmp_path, samples))
    assert len(events) == 5
    for ev, c in zip(events, centers):
        assert abs(ev.t_peak - c) < 0.03


def test_rest_produces_nothing():
    samples, _ = recording([], rest_s=3.0)
    assert detect_all(samples) == []


def test_rotation_without_acceleration_is_not_a_swing():
    samples, _ = recording([{"peak_g": 0.3, "peak_dps": 800}])
    assert detect_all(samples) == []


def test_bump_without_rotation_is_not_a_swing():
    samples, _ = recording([{"peak_g": 4, "peak_dps": 20}])
    assert detect_all(samples) == []


def test_double_peak_counts_once():
    samples, _ = recording([{"peak_g": 4, "peak_dps": 600, "double": True}] * 3)
    assert len(detect_all(samples)) == 3


def test_harder_swing_is_stronger_and_faster():
    samples, _ = recording([{"peak_g": 2.2, "peak_dps": 400}, {"peak_g": 5.5, "peak_dps": 900}])
    soft, hard = detect_all(samples)
    assert hard.strength > soft.strength
    assert hard.return_speed > soft.return_speed


def test_face_angle_sets_topspin_sign():
    # Default calibration: closed face = negative pitch = topspin.
    samples, _ = recording([{"peak_g": 4, "peak_dps": 600, "pitch_deg": -25},
                            {"peak_g": 4, "peak_dps": 600, "pitch_deg": 25}])
    closed, opened = detect_all(samples)
    assert closed.topspin > 0.3
    assert opened.topspin < -0.3


def test_yaw_change_sets_sidespin_sign():
    samples, _ = recording([{"peak_g": 4, "peak_dps": 600, "yaw_deg": 30},
                            {"peak_g": 4, "peak_dps": 600, "yaw_deg": -60}])
    right, left = detect_all(samples)
    assert right.sidespin > 0 > left.sidespin


def test_zero_from_rest_measures_scale():
    samples, _ = recording([], rest_s=1.0)
    cal = Calibration(accel_raw_per_g=1.0)
    cal.zero_from(samples)
    assert abs(cal.accel_raw_per_g - 1000) < 10
    assert abs(cal.gravity[2] - 1.0) < 0.01


RECORDINGS = sorted(glob.glob(os.path.join(HERE, "data", "*.csv")))


@pytest.mark.parametrize("path", RECORDINGS, ids=[os.path.basename(p) for p in RECORDINGS])
def test_recorded_csv(path):
    m = re.search(r"(\d+)swings", os.path.basename(path))
    if not m:
        pytest.skip("name the file like *_5swings.csv to set the expected count")
    cal = load_calibration(os.path.join(os.path.dirname(HERE), "swing_calibration.json"))
    assert len(detect_all(load_csv(path), cal)) == int(m.group(1))


def test_calibration_picks_spin_axis_and_sign():
    from tools.calibrate_swing import build_calibration, permissive

    rest, _ = recording([], rest_s=2.0)
    base = Calibration()
    probe = Calibration()
    probe.zero_from(rest)
    seg = permissive(probe)
    sw = lambda **kw: {"peak_g": 3.5, "peak_dps": 600, **kw}
    sets = {
        "soft": detect_all(recording([{"peak_g": 2.0, "peak_dps": 400}] * 5, seed=1)[0], seg),
        "hard": detect_all(recording([{"peak_g": 6.0, "peak_dps": 900}] * 5, seed=2)[0], seg),
        # This "mounting" puts the face angle on pitch with closed = POSITIVE pitch,
        # the opposite of the default guess.
        "closed": detect_all(recording([sw(pitch_deg=20 + i) for i in range(5)], seed=3)[0], seg),
        "open": detect_all(recording([sw(pitch_deg=-20 - i) for i in range(5)], seed=4)[0], seg),
        "right": detect_all(recording([sw(yaw_deg=-30 - i) for i in range(5)], seed=5)[0], seg),
        "left": detect_all(recording([sw(yaw_deg=30 + i) for i in range(5)], seed=6)[0], seg),
    }
    assert all(len(v) == 5 for v in sets.values()), {k: len(v) for k, v in sets.items()}
    cal = build_calibration(rest, sets, base)
    assert cal.topspin_terms[0]["feature"] == "pitch_at_peak"
    assert cal.strength_min_g < cal.strength_max_g
    closed = detect_all(recording([sw(pitch_deg=22)], seed=7)[0], cal)[0]
    opened = detect_all(recording([sw(pitch_deg=-22)], seed=8)[0], cal)[0]
    assert closed.topspin > 0.5 and opened.topspin < -0.5
    right = detect_all(recording([sw(yaw_deg=-32)], seed=9)[0], cal)[0]
    assert right.sidespin > 0.5
