#!/usr/bin/env python3
"""
tools/calibrate_swing.py -- guided swing calibration -> swing_calibration.json

Hold the Double Motor the way you will play with it, then follow the prompts:

  1. Hold still at the ready position for 3 s  -> neutral orientation, gravity, noise floor
  2. 5 soft forehand swings, then 5 hard ones   -> strength min/max (+ swing thresholds)
  3. 5 closed-face (topspin) swings, 5 open-face (backspin) swings
                                                -> which axis best separates them, sign and scale
  4. 5 swings brushing left, 5 brushing right   -> sidespin axis, sign and scale
  5. writes swing_calibration.json and prints a summary

Every step's raw IMU data is saved to imu_logs/calib_<step>.csv, so you can
re-run the math later without the hardware:

    python tools/calibrate_swing.py                 # live, with the paddle
    python tools/calibrate_swing.py --from-logs     # recompute from imu_logs/calib_*.csv

Recalibrate whenever the way the motor is held or mounted changes.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import threading
import time
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import config as C  # noqa: E402
from hardware.swing import Calibration, ImuSample, SwingEvent, detect_all, load_csv, save_csv  # noqa: E402

STEPS = [
    ("soft", "5 SOFT forehand swings"),
    ("hard", "5 HARD forehand swings"),
    ("closed", "5 swings with the paddle face CLOSED (tilted down, brushing up the back of the ball = topspin)"),
    ("open", "5 swings with the paddle face OPEN (tilted up, chopping under the ball = backspin)"),
    ("left", "5 swings BRUSHING LEFT across the ball (sidespin)"),
    ("right", "5 swings BRUSHING RIGHT across the ball (sidespin)"),
]
TOPSPIN_ANGLE_FEATURES = ["pitch_at_peak", "roll_at_peak", "yaw_at_peak"]
TOPSPIN_GYRO_FEATURES = ["gx_mean", "gy_mean", "gz_mean"]
SIDESPIN_FEATURES = ["yaw_delta", "gx_int", "gy_int", "gz_int", "roll_delta", "pitch_delta"]


# --------------------------------------------------------------------------
# The math (pure functions -- tested in tests/test_swing.py)
# --------------------------------------------------------------------------

def separation(a: Sequence[float], b: Sequence[float]) -> float:
    """How well a feature separates two groups: |mean difference| / pooled sd."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sd = math.sqrt((statistics.pvariance(a) + statistics.pvariance(b)) / 2) + 1e-6
    return abs(statistics.mean(a) - statistics.mean(b)) / sd


def spin_terms(positive: List[SwingEvent], negative: List[SwingEvent],
               feature_groups: List[List[str]]) -> List[dict]:
    """Pick the best-separating feature from each group and map the positive
    group's mean to +1 and the negative group's mean to -1. A group's term is
    dropped if it separates much worse than the best one."""
    picks = []
    for group in feature_groups:
        best = None
        for f in group:
            pa = [e.features[f] for e in positive if f in e.features]
            na = [e.features[f] for e in negative if f in e.features]
            d = separation(pa, na)
            if best is None or d > best[1]:
                best = (f, d, pa, na)
        if best and best[1] > 0:
            picks.append(best)
    if not picks:
        return []
    top = max(p[1] for p in picks)
    picks = [p for p in picks if p[1] >= 0.5 * top]
    total = sum(p[1] for p in picks)
    terms = []
    for f, d, pa, na in picks:
        mp, mn = statistics.mean(pa), statistics.mean(na)
        scale = (mp - mn) / 2 or 1.0
        terms.append({"feature": f, "center": (mp + mn) / 2, "scale": scale, "weight": d / total,
                      "separation": round(d, 2)})
    return terms


def build_calibration(rest: Sequence[ImuSample], sets: Dict[str, List[SwingEvent]],
                      base: Calibration) -> Calibration:
    cal = Calibration(**{k: getattr(base, k) for k in ("accel_raw_per_g", "gyro_raw_per_dps", "angle_raw_per_deg")})
    cal.zero_from(rest)
    soft, hard = sets.get("soft", []), sets.get("hard", [])
    all_swings = [e for v in sets.values() for e in v]
    if soft and hard:
        soft_peak = statistics.median(e.peak_accel_g for e in soft)
        hard_peak = statistics.median(e.peak_accel_g for e in hard)
        cal.strength_min_g = soft_peak * 0.8
        cal.strength_max_g = max(hard_peak, cal.strength_min_g + 0.5)
    if all_swings:
        # Trip at half of the gentlest swing, but well clear of the noise floor.
        cal.accel_threshold = max(5 * cal.noise_g, 0.3, 0.5 * min(e.peak_accel_g for e in all_swings))
        cal.gyro_threshold = max(5 * cal.noise_dps, 40.0, 0.5 * min(e.peak_gyro_dps for e in all_swings))
    if sets.get("closed") and sets.get("open"):
        cal.topspin_terms = spin_terms(sets["closed"], sets["open"],
                                       [TOPSPIN_ANGLE_FEATURES, TOPSPIN_GYRO_FEATURES]) or cal.topspin_terms
    if sets.get("right") and sets.get("left"):
        cal.sidespin_terms = spin_terms(sets["right"], sets["left"], [SIDESPIN_FEATURES]) or cal.sidespin_terms
    cal.source = "file"
    return cal


def permissive(cal: Calibration) -> Calibration:
    """Low thresholds for segmenting calibration swings."""
    p = Calibration(**{k: getattr(cal, k) for k in cal.__dataclass_fields__})
    p.accel_threshold = max(0.4, 6 * cal.noise_g)
    p.gyro_threshold = max(50.0, 6 * cal.noise_dps)
    return p


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        from hardware.paddle import Paddle
        self.paddle = Paddle(Calibration(), auto_zero=False)
        self.lock = threading.Lock()
        self.samples: List[ImuSample] = []
        self.recording = False
        self.paddle.raw_listeners.append(self._on)

    def _on(self, s: ImuSample) -> None:
        if self.recording:
            with self.lock:
                self.samples.append(s)

    def connect(self) -> bool:
        self.paddle.start()
        print("Connecting to the Double Motor...")
        for _ in range(150):
            if self.paddle.status == "connected":
                return True
            if self.paddle.status not in ("connecting", "disconnected"):
                break
            time.sleep(0.2)
        print(f"Paddle status: {self.paddle.status}")
        return False

    def record_until_enter(self) -> List[ImuSample]:
        with self.lock:
            self.samples = []
        self.recording = True
        input("   ...recording. Press Enter when done. ")
        self.recording = False
        with self.lock:
            return list(self.samples)

    def record_for(self, seconds: float) -> List[ImuSample]:
        with self.lock:
            self.samples = []
        self.recording = True
        time.sleep(seconds)
        self.recording = False
        with self.lock:
            return list(self.samples)


def log_path(step: str) -> str:
    return os.path.join(HERE, "imu_logs", f"calib_{step}.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-logs", action="store_true", help="recompute from imu_logs/calib_*.csv")
    ap.add_argument("--out", default=os.path.join(HERE, C.SWING_CALIBRATION_FILE))
    args = ap.parse_args()

    base = Calibration()
    rec = None
    if args.from_logs:
        rest = load_csv(log_path("rest"))
    else:
        rec = Recorder()
        if not rec.connect():
            return 1
        print("\nHold the paddle the way you will play.")
        input("Step 1: hold it STILL at your ready position, then press Enter (records 3 s). ")
        time.sleep(0.3)
        rest = rec.record_for(3.0)
        save_csv(log_path("rest"), rest)
    probe = Calibration(**{k: getattr(base, k) for k in base.__dataclass_fields__})
    probe.zero_from(rest)
    print(f"   rest: {len(rest)} samples, 1 g = {probe.accel_raw_per_g:.0f} raw, "
          f"noise {probe.noise_g:.3f} g / {probe.noise_dps:.1f} dps")
    seg = permissive(probe)

    sets: Dict[str, List[SwingEvent]] = {}
    for step, text in STEPS:
        if args.from_logs:
            samples = load_csv(log_path(step)) if os.path.exists(log_path(step)) else []
        else:
            input(f"\nNext: {text}. Press Enter to start. ")
            samples = rec.record_until_enter()
        events = detect_all(samples, seg)
        sets[step] = events
        if not args.from_logs:
            save_csv(log_path(step), samples)
            if len(events) == 5:
                save_csv(os.path.join(HERE, "imu_logs", f"calib_{step}_5swings.csv"), samples)
        flag = "" if len(events) == 5 else "   <-- expected 5; the result still uses what was found"
        print(f"   {step}: detected {len(events)} swings{flag}")
        if events:
            print("      peak accel g: " + ", ".join(f"{e.peak_accel_g:.1f}" for e in events)
                  + "   peak gyro dps: " + ", ".join(f"{e.peak_gyro_dps:.0f}" for e in events))

    if rec:
        rec.paddle.shutdown()
    cal = build_calibration(rest, sets, base)
    cal.save(args.out)

    print("\n=== Swing calibration ===")
    print(f"accel raw per g      {cal.accel_raw_per_g:.1f}")
    print("neutral angles (deg) " + ", ".join(f"{k} {v:.1f}" for k, v in cal.neutral.items()))
    print(f"thresholds           accel {cal.accel_threshold:.2f} g, gyro {cal.gyro_threshold:.0f} dps")
    print(f"strength range       {cal.strength_min_g:.2f} g (soft) .. {cal.strength_max_g:.2f} g (hard)")
    for name, terms in (("topspin", cal.topspin_terms), ("sidespin", cal.sidespin_terms)):
        for t in terms:
            print(f"{name:9s} term       {t['feature']:14s} center {t['center']:8.2f} scale {t['scale']:8.2f} "
                  f"weight {t['weight']:.2f}  (separation {t.get('separation', '-')})")
    weak = [t for t in cal.topspin_terms + cal.sidespin_terms if t.get("separation", 9) < 1.0]
    if weak:
        print("Warning: some spin features separate poorly (< 1.0) -- exaggerate the face angle / brushing and recalibrate.")
    # Sanity: re-score the spin swings with the new calibration.
    for step, expect in (("closed", "+"), ("open", "-"), ("right", "+"), ("left", "-")):
        events = detect_all(load_csv(log_path(step)), cal) if os.path.exists(log_path(step)) else []
        key = "topspin" if step in ("closed", "open") else "sidespin"
        vals = [getattr(e, key) for e in events]
        if vals:
            print(f"check {step:6s} -> {key} " + ", ".join(f"{v:+.2f}" for v in vals) + f"   (expect {expect})")
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
