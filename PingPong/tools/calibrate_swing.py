#!/usr/bin/env python3
"""
tools/calibrate_swing.py -- guided swing calibration -> swing_calibration.json

Hold the Double Motor the way you will play with it, then follow the prompts.
Each swing/toss step asks for N repetitions (--swings, default 5; ~10 gives more
reliable spin and forehand/backhand detection):

  1. Hold still at the ready position for 3 s  -> neutral orientation, gravity, gyro bias, noise
  2. Slowly tilt the face DOWN ~45 deg and hold -> gyro scale + how the motor's axes map onto the
                                                   on-screen paddle
  3. N serve TOSSES: sharp upward flicks       -> how hard / straight / twist-free your toss is,
                                                   tuned so none of your swings count as a toss
  4. N soft + N hard FOREHANDS, then N soft + N hard BACKHANDS
                                                -> strength min/max, swing thresholds, and how to
                                                   tell a forehand from a backhand by its rotation
  5. N closed-face + N open-face forehands, same for backhands
                                                -> topspin/backspin axis, sign and scale -- separately
                                                   for each stroke, since a backhand flips the face
  6. N swings brushing left, N brushing right   -> sidespin axis, sign and scale
  7. writes swing_calibration.json and prints a summary

Every step's raw IMU data is saved to imu_logs/calib_<step>.csv, so you can
re-run the math later without the hardware:

    python tools/calibrate_swing.py                 # live, with the paddle
    python tools/calibrate_swing.py --swings 10     # 10 of each instead of 5
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
from hardware.swing import (Calibration, ImuSample, SwingDetector, SwingEvent, TossDetector, detect_all,
                            load_csv, save_csv)  # noqa: E402

STEPS = [
    ("soft", "{n} SOFT FOREHAND swings"),
    ("hard", "{n} HARD FOREHAND swings"),
    ("bh_soft", "{n} SOFT BACKHAND swings"),
    ("bh_hard", "{n} HARD BACKHAND swings"),
    ("closed", "{n} FOREHANDS with the face CLOSED (tilted down, brushing up the back of the ball = topspin)"),
    ("open", "{n} FOREHANDS with the face OPEN (tilted up, chopping under the ball = backspin)"),
    ("bh_closed", "{n} BACKHANDS with the face CLOSED (topspin)"),
    ("bh_open", "{n} BACKHANDS with the face OPEN (backspin)"),
    ("left", "{n} swings BRUSHING LEFT across the ball (sidespin)"),
    ("right", "{n} swings BRUSHING RIGHT across the ball (sidespin)"),
]
FOREHAND_SETS = ("soft", "hard", "closed", "open")
BACKHAND_SETS = ("bh_soft", "bh_hard", "bh_closed", "bh_open")
STROKE_FEATURES = ["gx_mean", "gy_mean", "gz_mean", "gx_int", "gy_int", "gz_int", "yaw_delta"]
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


def analyze_tilt(rest: Sequence[ImuSample], tilt: Sequence[ImuSample], cal: Calibration):
    """From a slow face-down tilt that ends held still: the gyro's raw counts per
    deg/s (integrated rotation vs. the change in gravity direction) and the
    motor-frame axis of that rotation (the "close the face" axis). Returns
    (raw_per_dps, axis, angle_deg) or None if the tilt was too small."""
    if len(rest) < 5 or len(tilt) < 20:
        return None
    n = len(rest)
    bias = [sum(getattr(s, k) for s in rest) / n for k in ("gx", "gy", "gz")]
    t_end = tilt[-1].t
    # "Before" = the samples ahead of any rotation (or the rest recording).
    sd = max(1e-6, max(math.sqrt(sum((getattr(s, k) - bias[i]) ** 2 for i, k in enumerate(("gx", "gy", "gz"))))
                       for s in rest))
    start = []
    for s in tilt:
        if math.sqrt(sum((getattr(s, k) - bias[i]) ** 2 for i, k in enumerate(("gx", "gy", "gz")))) > 3 * sd:
            break
        start.append(s)
    start = start or rest
    held = [s for s in tilt if t_end - s.t < 0.4]
    g0 = [sum(getattr(s, k) for s in start) / len(start) for k in ("ax", "ay", "az")]
    g1 = [sum(getattr(s, k) for s in held) / len(held) for k in ("ax", "ay", "az")]
    c = sum(a * b for a, b in zip(g0, g1)) / (math.sqrt(sum(a * a for a in g0)) * math.sqrt(sum(b * b for b in g1)))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    if angle < 15:
        return None
    integ = [0.0, 0.0, 0.0]
    for a, b in zip(tilt, tilt[1:]):
        dt = b.t - a.t
        for i, k in enumerate(("gx", "gy", "gz")):
            integ[i] += (getattr(a, k) - bias[i]) * dt
    mag = math.sqrt(sum(x * x for x in integ))
    if mag < 1e-9:
        return None
    return mag / angle, [x / mag for x in integ], angle


def build_calibration(rest: Sequence[ImuSample], sets: Dict[str, List[SwingEvent]],
                      base: Calibration) -> Calibration:
    cal = Calibration(**{k: getattr(base, k) for k in ("accel_raw_per_g", "gyro_raw_per_dps", "angle_raw_per_deg",
                                                        "close_axis")})
    cal.zero_from(rest)
    soft = sets.get("soft", []) + sets.get("bh_soft", [])
    hard = sets.get("hard", []) + sets.get("bh_hard", [])
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
    if sets.get("bh_closed") and sets.get("bh_open"):
        cal.topspin_terms_backhand = spin_terms(sets["bh_closed"], sets["bh_open"],
                                                [TOPSPIN_ANGLE_FEATURES, TOPSPIN_GYRO_FEATURES]) or None
    forehands = [e for k in FOREHAND_SETS for e in sets.get(k, [])]
    backhands = [e for k in BACKHAND_SETS for e in sets.get(k, [])]
    if len(forehands) >= 3 and len(backhands) >= 3:
        # One feature that tells the strokes apart (> 0 = forehand), e.g. the
        # sign of the rotation about the swing axis.
        terms = spin_terms(forehands, backhands, [STROKE_FEATURES])
        cal.stroke_terms = terms[:1] or None
    if sets.get("right") and sets.get("left"):
        cal.sidespin_terms = spin_terms(sets["right"], sets["left"], [SIDESPIN_FEATURES]) or cal.sidespin_terms
    cal.source = "file"
    return cal


def toss_candidates(samples: Sequence[ImuSample], cal: Calibration) -> List[dict]:
    """Every upward jolt in a recording, measured with wide-open toss thresholds."""
    probe = Calibration(**{k: getattr(cal, k) for k in cal.__dataclass_fields__})
    probe.toss_accel_g, probe.toss_max_gyro_dps, probe.toss_up_fraction = 0.3, 1e9, 0.0
    sd = SwingDetector(probe)
    td = TossDetector(sd, record=True)
    for smp in samples:
        sd.push(smp)
        td.push(smp)
    return td.candidates


def count_tosses(samples: Sequence[ImuSample], cal: Calibration) -> int:
    sd = SwingDetector(cal)
    td = TossDetector(sd)
    n = 0
    for smp in samples:
        sd.push(smp)
        n += td.push(smp) is not None
    return n


def calibrate_toss(cal: Calibration, toss_samples: Sequence[ImuSample],
                   swing_samples: Sequence[Sequence[ImuSample]]) -> List[str]:
    """Set the toss thresholds on `cal` from the recorded flicks, keeping every
    recorded swing below them. Returns warnings (empty if all is well)."""
    warnings = []
    cands = toss_candidates(toss_samples, cal)
    if not cands:
        return ["toss: no upward flicks found -- flick the paddle sharply straight up"]
    top = max(c["peak_g"] for c in cands)
    tosses = [c for c in cands if c["peak_g"] >= 0.4 * top]       # every clear flick
    if len(tosses) < 2:
        return [f"toss: only {len(tosses)} clear flick(s) found -- keeping the default toss settings"]
    peaks = [c["peak_g"] for c in tosses]
    cal.toss_accel_g = max(0.3, 4 * cal.noise_g, 0.6 * min(peaks))
    cal.toss_up_fraction = max(0.35, 0.85 * min(c["up_frac"] for c in tosses))
    cal.toss_peak_min_g, cal.toss_peak_max_g = min(peaks), max(peaks)
    toss_gyro = max(c["gyro_dps"] for c in tosses)
    # Swing jolts that would pass the accel + direction tests: only rotation can tell them apart.
    rivals = [c["gyro_dps"] for rec in swing_samples for c in toss_candidates(rec, cal)
              if c["peak_g"] >= cal.toss_accel_g and c["up_frac"] >= cal.toss_up_fraction]
    if not rivals:
        cal.toss_max_gyro_dps = toss_gyro * 1.4 + 20
    elif min(rivals) > toss_gyro:
        cal.toss_max_gyro_dps = (toss_gyro + min(rivals)) / 2
    else:
        cal.toss_max_gyro_dps = toss_gyro * 1.1 + 10
        warnings.append("toss: some swings look like tosses (as upward and as twist-free); keep the toss flick "
                        "straight up and the swings sweeping")
    return warnings


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
    ap.add_argument("--swings", type=int, default=5, metavar="N",
                    help="repetitions per swing/toss step (default 5; 3-30). More = more reliable spin and "
                         "forehand/backhand detection; ~10 is plenty")
    args = ap.parse_args()
    n = args.swings
    if not 3 <= n <= 30:
        ap.error("--swings must be between 3 and 30")

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
    if args.from_logs:
        tilt = load_csv(log_path("tilt")) if os.path.exists(log_path("tilt")) else []
    else:
        input("\nStep 2: from the ready position, SLOWLY tilt the paddle face DOWN toward the table\n"
              "        (about 45 degrees) and HOLD it there. Press Enter to start, Enter again once held. ")
        tilt = rec.record_until_enter()
        save_csv(log_path("tilt"), tilt)
    tilt_result = analyze_tilt(rest, tilt, base)
    if tilt_result:
        base.gyro_raw_per_dps, base.close_axis, angle = tilt_result
        print(f"   tilt: {angle:.0f} deg -> gyro {base.gyro_raw_per_dps:.2f} raw per deg/s, "
              f"face-closing axis {[round(a, 2) for a in base.close_axis]}")
    else:
        print("   tilt: too small or missing -- keeping the default gyro scale and a guessed paddle axis.\n"
              "         (The on-screen paddle may turn the wrong way; rerun and tilt further.)")
    if args.from_logs:
        toss = load_csv(log_path("toss")) if os.path.exists(log_path("toss")) else []
    else:
        input(f"\nStep 3: {n} serve TOSSES. From your ready position, flick the paddle sharply STRAIGHT UP\n"
              "        (as if tossing the ball), with a pause between flicks. Press Enter to start. ")
        toss = rec.record_until_enter()
        save_csv(log_path("toss"), toss)
    probe = Calibration(**{k: getattr(base, k) for k in base.__dataclass_fields__})
    probe.zero_from(rest)
    print(f"   rest: {len(rest)} samples, 1 g = {probe.accel_raw_per_g:.0f} raw, "
          f"noise {probe.noise_g:.3f} g / {probe.noise_dps:.1f} dps")
    seg = permissive(probe)

    sets: Dict[str, List[SwingEvent]] = {}
    swing_samples: Dict[str, List[ImuSample]] = {}
    for step, text in STEPS:
        if args.from_logs:
            samples = load_csv(log_path(step)) if os.path.exists(log_path(step)) else []
        else:
            input(f"\nNext: {text.format(n=n)}. Press Enter to start. ")
            samples = rec.record_until_enter()
        events = detect_all(samples, seg)
        sets[step] = events
        swing_samples[step] = samples
        if not args.from_logs:
            save_csv(log_path(step), samples)
            if len(events) == n:
                save_csv(os.path.join(HERE, "imu_logs", f"calib_{step}_{n}swings.csv"), samples)
        flag = "" if len(events) == n or args.from_logs else f"   <-- expected {n}; the result still uses what was found"
        print(f"   {step}: detected {len(events)} swings{flag}")
        if events:
            print("      peak accel g: " + ", ".join(f"{e.peak_accel_g:.1f}" for e in events)
                  + "   peak gyro dps: " + ", ".join(f"{e.peak_gyro_dps:.0f}" for e in events))

    if rec:
        rec.paddle.shutdown()
    cal = build_calibration(rest, sets, base)
    toss_warnings = calibrate_toss(cal, toss, list(swing_samples.values())) if toss else \
        ["toss: no toss recording -- keeping the default toss settings"]
    cal.save(args.out)

    print("\n=== Swing calibration ===")
    print(f"accel raw per g      {cal.accel_raw_per_g:.1f}")
    print("neutral angles (deg) " + ", ".join(f"{k} {v:.1f}" for k, v in cal.neutral.items()))
    print(f"thresholds           accel {cal.accel_threshold:.2f} g, gyro {cal.gyro_threshold:.0f} dps")
    print(f"strength range       {cal.strength_min_g:.2f} g (soft) .. {cal.strength_max_g:.2f} g (hard)")
    print(f"serve toss           >= {cal.toss_accel_g:.2f} g upward, <= {cal.toss_max_gyro_dps:.0f} deg/s rotation, "
          f">= {cal.toss_up_fraction:.0%} straight up")
    if toss:
        print(f"check toss        -> {count_tosses(toss, cal)} flicks count as tosses (asked for {n}); "
              f"{sum(count_tosses(v, cal) for v in swing_samples.values())} of your swings do (should be 0)")
    for w in toss_warnings:
        print("Warning: " + w)
    for name, terms in (("topspin FH", cal.topspin_terms), ("topspin BH", cal.topspin_terms_backhand or []),
                        ("sidespin", cal.sidespin_terms), ("FH vs BH", cal.stroke_terms or [])):
        for t in terms:
            print(f"{name:9s} term       {t['feature']:14s} center {t['center']:8.2f} scale {t['scale']:8.2f} "
                  f"weight {t['weight']:.2f}  (separation {t.get('separation', '-')})")
    weak = [t for t in cal.topspin_terms + (cal.topspin_terms_backhand or []) + cal.sidespin_terms
            + (cal.stroke_terms or []) if t.get("separation", 9) < 1.0]
    if weak:
        print("Warning: some spin features separate poorly (< 1.0) -- exaggerate the face angle / brushing and recalibrate.")
    # Sanity: re-score the spin swings with the new calibration.
    for step, expect in (("closed", "+"), ("open", "-"), ("bh_closed", "+"), ("bh_open", "-"),
                         ("right", "+"), ("left", "-")):
        events = detect_all(load_csv(log_path(step)), cal) if os.path.exists(log_path(step)) else []
        key = "topspin" if "closed" in step or "open" in step else "sidespin"
        vals = [getattr(e, key) for e in events]
        if vals:
            print(f"check {step:9s} -> {key} " + ", ".join(f"{v:+.2f}" for v in vals) + f"   (expect {expect})")
    for step, expect in (("soft", "forehand"), ("bh_soft", "backhand")):
        events = detect_all(load_csv(log_path(step)), cal) if os.path.exists(log_path(step)) else []
        if events and cal.stroke_terms:
            ok = sum(e.imu_stroke == expect for e in events)
            print(f"check {step:9s} -> read as {expect} {ok}/{len(events)}")
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
