"""Swing classifiers, recovery suppression and calibration-set cleaning."""

import random

from hardware import swing_model
from hardware.swing import Calibration, SwingEvent, detect_all
from tests.synth import recording
from tools.calibrate_swing import clean_set, permissive


def fake(feature_value, noise_rng, label_shift=0.0):
    """Feature dict: one informative feature plus noise features."""
    f = {n: noise_rng.gauss(0, 0.01 if n.endswith(("x_pk", "y_pk", "z_pk")) and n[0] == "l" else 3)
         for n in swing_model.FEATURES}
    f["pitch_at_peak"] = feature_value
    return f


def test_classifier_learns_the_informative_feature_and_scores_honestly():
    rng = random.Random(0)
    pos = [fake(20 + rng.gauss(0, 3), rng) for _ in range(12)]
    neg = [fake(-20 + rng.gauss(0, 3), rng) for _ in range(12)]
    m = swing_model.fit(pos, neg)
    assert m["loo_accuracy"] == 1.0
    assert swing_model.signed(m, fake(22, rng)) > 0.5
    assert swing_model.signed(m, fake(-22, rng)) < -0.5
    assert abs(swing_model.signed(m, fake(0, rng))) < 0.5          # in between -> weak


def test_classifier_does_not_invent_patterns_in_pure_noise():
    rng = random.Random(1)
    pos = [fake(rng.gauss(0, 3), rng) for _ in range(10)]
    neg = [fake(rng.gauss(0, 3), rng) for _ in range(10)]
    m = swing_model.fit(pos, neg)
    assert m["loo_accuracy"] < 0.8                                    # can't beat chance by much


def test_recovery_motion_after_a_swing_is_not_a_second_swing():
    # Each swing followed 0.45 s later by a weaker return-to-ready motion.
    swings = []
    for i in range(4):
        swings += [{"peak_g": 4, "peak_dps": 600}, {"peak_g": 1.6, "peak_dps": -350}]
    samples, centers = recording(swings, gap_s=0.45)
    events = detect_all(samples)
    assert len(events) == 4
    assert all(e.peak_accel_g > 3 for e in events)


def test_clean_set_merges_doubles_and_drops_startup_jolts():
    mk = lambda t, g, w: SwingEvent(t_peak=t, strength=0.5, return_speed=7, topspin=0, sidespin=0,  # noqa: E731
                                    peak_accel_g=g, peak_gyro_dps=w)
    events = [mk(0.0, 1.6, 60),                                  # start-up jolt: barely rotating
              mk(1.0, 3.0, 450), mk(1.3, 3.4, 470),               # one swing caught twice
              mk(2.0, 2.9, 430), mk(3.0, 3.1, 440), mk(4.0, 3.0, 420)]
    cleaned = clean_set(events)
    assert [e.t_peak for e in cleaned] == [1.3, 2.0, 3.0, 4.0]


def test_quick_weaker_retry_counts_when_it_is_a_forward_stroke():
    """After a too-early whiff, a hurried second swing may be weaker than the first.
    It must not be thrown away as a 'return to ready' if it's a real forward stroke."""
    class Tracker:                       # stand-in for hardware.stroke.StrokeTracker
        def __init__(self, mode): self.mode = mode
        def mode_at(self, t): return self.mode
        def drawing_back(self, t): return False
    samples, _ = recording([{"peak_g": 4.0, "peak_dps": 600}, {"peak_g": 2.8, "peak_dps": 500}], gap_s=0.5)
    from hardware.swing import SwingDetector
    for mode, expected in (("forward", 2), ("settle", 1)):
        det = SwingDetector(Calibration())
        det.stroke = Tracker(mode)
        evs = [e for e in (det.push(s) for s in samples) if e]
        assert len(evs) == expected, (mode, [e.peak_accel_g for e in evs])
