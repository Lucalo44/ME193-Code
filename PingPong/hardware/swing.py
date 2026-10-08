"""
hardware/swing.py -- turn a stream of Double Motor IMU samples into swing
events carrying return speed and spin.

IMU units
---------
legoeducation 1.1.1 reports raw int16 values (ImuDeviceNotification) and does
not document their units. What we rely on:

  * Accelerometer: the at-rest magnitude is exactly 1 g, so zero() measures
    raw-counts-per-g directly (Calibration.accel_raw_per_g). Until then
    DEFAULT_ACCEL_RAW_PER_G (1000, i.e. milli-g, as on SPIKE hubs) is assumed.
  * Gyro: assumed deg/s (DEFAULT_GYRO_RAW_PER_DPS = 1). Check with
    tools/imu_logger.py: rotate the paddle exactly 90 degrees about one axis
    and read the integrated-gyro summary it prints; counts / 90 = raw per dps.
  * yaw/pitch/roll: assumed decidegrees (ANGLE_RAW_PER_DEG = 10). Check by
    tilting the paddle 90 degrees in imu_logger.py.

  Measured on our hardware (fill in after Milestone 2):
      accel raw per g   = ____
      gyro raw per dps  = ____
      angle raw per deg = ____
      sample interval   = ____ ms (asked for 15)

Detection
---------
Linear acceleration = accel (in g) minus a gravity estimate that is captured
at zero() and slowly re-learned whenever the paddle is still. A swing event
fires when gyro magnitude exceeds the gyro threshold and linear acceleration
exceeds the accel threshold within SWING_COINCIDENCE_S of each other. The
swing ends after both signals drop below threshold * SWING_HYSTERESIS for
SWING_END_QUIET_S. The event is reported as soon as the strike is over --
acceleration back below SWING_PAST_PEAK_RATIO of its peak, at least the strike
window after it -- rather than when the follow-through settles. On our
recordings that cut the median reporting delay from ~260 ms to ~110 ms.
One event is emitted per swing, and a SWING_COOLDOWN_S
refractory period suppresses double triggers. t_peak is the time of peak
linear acceleration.

Serve toss
----------
TossDetector watches the same stream for the serve toss: a sharp jolt of the
paddle straight UP (linear acceleration along the at-rest gravity direction,
above TOSS_ACCEL_G) with little rotation (gyro below TOSS_MAX_GYRO_DPS), so it
isn't confused with a swing. The jolt's strength sets the toss height.

Spin
----
Which axes mean "face angle" and "brushing direction" depends on how the motor
is held, so nothing is hard-coded: spin is a weighted sum of named features
(see swing_features) whose centers/scales/weights come from
tools/calibrate_swing.py and are stored in swing_calibration.json.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import config as C
from hardware import swing_model


@dataclass
class ImuSample:
    t: float                      # time.monotonic() on arrival
    yaw: float
    pitch: float
    roll: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


CSV_FIELDS = ["t", "yaw", "pitch", "roll", "ax", "ay", "az", "gx", "gy", "gz"]


class SampleClock:
    """Bluetooth delivers the motor's samples in bursts (half arrive <5 ms after
    the previous one, with 30-300 ms gaps between bursts) although the motor
    takes one every ~15 ms. Timestamping on arrival scrambles anything that
    integrates over time, so this rebuilds an even clock: each sample is one
    period after the previous, never later than its arrival and never more than
    MAX_LAG_S before it. The period is learned from the arrival rate."""

    MAX_LAG_S = 0.4
    WINDOW = 200

    def __init__(self, period_s: float = C.IMU_NOTIFICATION_MS / 1000.0):
        self.period = period_s
        self._arrivals: Deque[float] = deque(maxlen=self.WINDOW)
        self._t: Optional[float] = None

    def stamp(self, arrival: float) -> float:
        self._arrivals.append(arrival)
        if len(self._arrivals) >= 50:
            self.period = (self._arrivals[-1] - self._arrivals[0]) / (len(self._arrivals) - 1)
        if self._t is None:
            self._t = arrival
        else:
            self._t = min(arrival, max(self._t + self.period, arrival - self.MAX_LAG_S))
        return self._t


def retime(samples: Sequence["ImuSample"]) -> List["ImuSample"]:
    """Re-time a recording made with arrival timestamps (offline version of SampleClock)."""
    from dataclasses import replace
    if len(samples) < 2:
        return list(samples)
    period = (samples[-1].t - samples[0].t) / (len(samples) - 1)
    # Offline we can see the whole recording: walk back from the end so each
    # sample is one period before the next, never later than its own arrival
    # and never more than MAX_LAG_S before it.
    t = [s.t for s in samples]
    for i in range(len(t) - 2, -1, -1):
        t[i] = max(min(samples[i].t, t[i + 1] - period), samples[i].t - SampleClock.MAX_LAG_S)
    return [replace(s, t=ti) for s, ti in zip(samples, t)]


def save_csv(path: str, samples: Sequence[ImuSample]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for s in samples:
            w.writerow([f"{s.t:.4f}"] + [int(getattr(s, k)) for k in CSV_FIELDS[1:]])


def load_csv(path: str) -> List[ImuSample]:
    with open(path, newline="") as f:
        return [ImuSample(**{k: float(row[k]) for k in CSV_FIELDS}) for row in csv.DictReader(f)]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def _default_topspin_terms() -> list:
    # Closed face assumed to be negative pitch. calibrate_swing.py replaces this.
    return [{"feature": "pitch_at_peak", "center": 0.0, "scale": -30.0, "weight": 1.0}]


def _default_sidespin_terms() -> list:
    return [{"feature": "yaw_delta", "center": 0.0, "scale": 40.0, "weight": 1.0}]


@dataclass
class Calibration:
    accel_raw_per_g: float = C.DEFAULT_ACCEL_RAW_PER_G
    gyro_raw_per_dps: float = C.DEFAULT_GYRO_RAW_PER_DPS
    angle_raw_per_deg: float = C.ANGLE_RAW_PER_DEG
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])   # in g, sensor frame
    neutral: Dict[str, float] = field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    noise_g: float = 0.05
    noise_dps: float = 5.0
    accel_threshold: float = C.SWING_ACCEL_THRESHOLD
    gyro_threshold: float = C.SWING_GYRO_THRESHOLD
    strength_min_g: float = C.DEFAULT_STRENGTH_MIN_G
    strength_max_g: float = C.DEFAULT_STRENGTH_MAX_G
    topspin_terms: list = field(default_factory=_default_topspin_terms)       # forehand (or all strokes)
    sidespin_terms: list = field(default_factory=_default_sidespin_terms)
    # Backhands flip the paddle face, so they get their own topspin terms, chosen
    # by stroke_terms (> 0 = forehand) -- both learned by calibrate_swing.py.
    topspin_terms_backhand: Optional[list] = None
    stroke_terms: Optional[list] = None
    gyro_bias: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])   # raw, measured at rest
    # Motor-frame axis the paddle rotates about when its face closes; maps the
    # motor onto the on-screen paddle (see hardware/orientation.py).
    close_axis: Optional[List[float]] = None
    # Serve toss (an upward flick), learned by the calibration tool's toss step.
    toss_accel_g: float = C.TOSS_ACCEL_G
    toss_max_gyro_dps: float = C.TOSS_MAX_GYRO_DPS
    toss_up_fraction: float = C.TOSS_UP_FRACTION
    toss_peak_min_g: Optional[float] = None     # gentlest / strongest calibration flick,
    toss_peak_max_g: Optional[float] = None     # mapped to the lowest / highest toss
    # Learned classifiers (hardware/swing_model.py), keyed "stroke", "topspin_fh",
    # "topspin_bh", "sidespin". Each is used only if it beat the single-feature
    # rule above in leave-one-out testing ("use": true).
    models: Optional[dict] = None
    # Signed on-screen yaw of this player's full forehand / backhand backswing
    # (hardware/stroke.py); learned from the calibration swings.
    stroke_back_fh_deg: Optional[float] = None
    stroke_back_bh_deg: Optional[float] = None
    source: str = "default"

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        cal = cls(**known)
        cal.source = "file"
        return cal

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def angles_deg(self, s: ImuSample) -> Dict[str, float]:
        k = self.angle_raw_per_deg
        return {"yaw": s.yaw / k, "pitch": s.pitch / k, "roll": s.roll / k}

    def relative_angles(self, s: ImuSample) -> Dict[str, float]:
        a = self.angles_deg(s)
        return {name: wrap_deg(a[name] - self.neutral.get(name, 0.0)) for name in a}

    def zero_from(self, still: Sequence[ImuSample]) -> None:
        """Capture the ready position from samples of a motionless paddle:
        accel scale (1 g), gravity direction, neutral angles, noise floor."""
        if not still:
            return
        n = len(still)
        mean = [sum(getattr(s, k) for s in still) / n for k in ("ax", "ay", "az")]
        mag = math.sqrt(sum(m * m for m in mean))
        if mag > 1e-6:
            self.accel_raw_per_g = mag
            self.gravity = [m / mag for m in mean]
        angles = [self.angles_deg(s) for s in still]
        self.neutral = {k: sum(a[k] for a in angles) / n for k in ("yaw", "pitch", "roll")}
        self.gyro_bias = [sum(getattr(s, k) for s in still) / n for k in ("gx", "gy", "gz")]
        lin = [linear_g(s, self, self.gravity) for s in still]
        gyr = [gyro_dps(s, self) for s in still]
        self.noise_g = max(lin) if lin else self.noise_g
        self.noise_dps = max(gyr) if gyr else self.noise_dps


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def accel_g(s: ImuSample, cal: Calibration) -> tuple:
    k = cal.accel_raw_per_g
    return (s.ax / k, s.ay / k, s.az / k)


def linear_g(s: ImuSample, cal: Calibration, gravity: Sequence[float]) -> float:
    a = accel_g(s, cal)
    return math.sqrt(sum((a[i] - gravity[i]) ** 2 for i in range(3)))


def gyro_vec_dps(s: ImuSample, cal: Calibration) -> tuple:
    k, b = cal.gyro_raw_per_dps, cal.gyro_bias
    return ((s.gx - b[0]) / k, (s.gy - b[1]) / k, (s.gz - b[2]) / k)


def gyro_dps(s: ImuSample, cal: Calibration) -> float:
    g = gyro_vec_dps(s, cal)
    return math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])


# --------------------------------------------------------------------------
# Swing events
# --------------------------------------------------------------------------

@dataclass
class SwingEvent:
    t_peak: float
    strength: float
    return_speed: float
    topspin: float
    sidespin: float
    peak_accel_g: float
    peak_gyro_dps: float
    features: Dict[str, float] = field(default_factory=dict)
    raw: List[ImuSample] = field(default_factory=list)
    stroke_hint: Optional[str] = None   # sim paddle only: forced stroke for --no-camera testing
    imu_stroke: Optional[str] = None    # forehand/backhand as told by the IMU (needs calibration)

    def trace(self, cal: Calibration) -> list:
        """Downsampled [t_rel, linear_g, gyro_dps] rows for the HUD swing trace."""
        g = cal.gravity
        return [[round(s.t - self.t_peak, 3), round(linear_g(s, cal, g), 2), round(gyro_dps(s, cal), 1)]
                for s in self.raw]

    def summary(self) -> dict:
        return {
            "strength": round(self.strength, 3),
            "return_speed": round(self.return_speed, 2),
            "topspin": round(self.topspin, 3),
            "sidespin": round(self.sidespin, 3),
            "peak_accel_g": round(self.peak_accel_g, 2),
            "peak_gyro_dps": round(self.peak_gyro_dps, 1),
            "imu_stroke": self.imu_stroke,
        }


def strength_to_speed(strength: float) -> float:
    s = max(0.0, min(1.0, strength)) ** C.RETURN_STRENGTH_CURVE
    return C.RETURN_SPEED_MIN + (C.RETURN_SPEED_MAX - C.RETURN_SPEED_MIN) * s


def swing_features(window: Sequence[ImuSample], t_peak: float, cal: Calibration,
                   gravity: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Features of one swing, measured around the strike -- the moment of peak
    acceleration -- not over the whole motion, so the backswing and the return
    to ready don't cancel the stroke out. Angles are in degrees relative to the
    neutral (zeroed) orientation; gyro values in deg/s and degrees.

    *_pk are the rotation rates and the direction of linear acceleration right
    at the strike: the most telling inputs for the swing classifier."""
    before, after = C.SWING_STRIKE_WINDOW_S
    strike = [s for s in window if t_peak - before <= s.t <= t_peak + after]
    window = strike if len(strike) >= 3 else window
    if not window:
        return {}
    at_peak = min(window, key=lambda s: abs(s.t - t_peak))
    rel = cal.relative_angles(at_peak)
    first, last = cal.angles_deg(window[0]), cal.angles_deg(window[-1])
    feats = {
        "pitch_at_peak": rel["pitch"],
        "roll_at_peak": rel["roll"],
        "yaw_at_peak": rel["yaw"],
        "yaw_delta": wrap_deg(last["yaw"] - first["yaw"]),
        "pitch_delta": wrap_deg(last["pitch"] - first["pitch"]),
        "roll_delta": wrap_deg(last["roll"] - first["roll"]),
    }
    k = cal.gyro_raw_per_dps
    for axis in ("gx", "gy", "gz"):
        vals = [getattr(s, axis) / k for s in window]
        feats[f"{axis}_mean"] = sum(vals) / len(vals)
        integral = 0.0
        for a, b in zip(window, window[1:]):
            integral += getattr(a, axis) / k * (b.t - a.t)
        feats[f"{axis}_int"] = integral
    near = [s for s in window if abs(s.t - t_peak) <= 0.03] or [at_peak]
    for axis in ("gx", "gy", "gz"):
        feats[f"{axis}_pk"] = sum(getattr(s, axis) for s in near) / len(near) / k
    g = gravity if gravity is not None else cal.gravity
    a = accel_g(at_peak, cal)
    lin = [a[i] - g[i] for i in range(3)]
    mag = math.sqrt(sum(x * x for x in lin)) or 1e-9
    for i, axis in enumerate(("lx", "ly", "lz")):
        feats[f"{axis}_pk"] = lin[i] / mag
    return feats


def combine_terms(terms: Sequence[dict], feats: Dict[str, float]) -> float:
    total = 0.0
    for term in terms:
        f = feats.get(term["feature"])
        if f is None or not term.get("scale"):
            continue
        total += term.get("weight", 1.0) * (f - term.get("center", 0.0)) / term["scale"]
    return max(-1.0, min(1.0, total))


class SwingDetector:
    PRE_ROLL_S = 0.15

    def __init__(self, cal: Optional[Calibration] = None):
        self.cal = cal or Calibration()
        self.gravity = list(self.cal.gravity)
        self._recent: Deque[ImuSample] = deque()
        self._active = False
        self._window: List[ImuSample] = []
        self._last_event_t = -math.inf
        self._last_event_g = 0.0
        # Optional live stroke tracker (hardware/stroke.py): when set, a burst of
        # motion while the paddle is being drawn back is not reported as a swing.
        self.stroke = None
        self._reset_swing()

    def set_calibration(self, cal: Calibration) -> None:
        self.cal = cal
        self.gravity = list(cal.gravity)

    def _reset_swing(self) -> None:
        self._start = 0.0
        self._peak_lin = 0.0
        self._peak_lin_t = 0.0
        self._peak_gyro = 0.0
        self._last_gyro_trip = -math.inf
        self._last_accel_trip = -math.inf
        self._coincided = False
        self._spent = False           # this swing has already been reported
        self._quiet_since: Optional[float] = None

    def push(self, s: ImuSample) -> Optional[SwingEvent]:
        cal = self.cal
        lin = linear_g(s, cal, self.gravity)
        gyr = gyro_dps(s, cal)
        a_thr, g_thr = cal.accel_threshold, cal.gyro_threshold

        self._recent.append(s)
        while self._recent and s.t - self._recent[0].t > self.PRE_ROLL_S:
            self._recent.popleft()

        if not self._active:
            # Re-learn gravity while still, so slow drift in hold angle
            # doesn't look like linear acceleration.
            if gyr < max(30.0, 3 * cal.noise_dps):
                a = accel_g(s, cal)
                mag = math.sqrt(sum(x * x for x in a))
                if abs(mag - 1.0) < 0.1:
                    self.gravity = [0.97 * g + 0.03 * x for g, x in zip(self.gravity, a)]
            if gyr > g_thr or lin > a_thr:
                self._active = True
                self._reset_swing()
                self._start = s.t
                self._window = list(self._recent)
            else:
                return None
        else:
            self._window.append(s)

        if lin > self._peak_lin:
            self._peak_lin, self._peak_lin_t = lin, s.t
        self._peak_gyro = max(self._peak_gyro, gyr)
        if gyr > g_thr:
            self._last_gyro_trip = s.t
            if s.t - self._last_accel_trip <= C.SWING_COINCIDENCE_S:
                self._coincided = True
        if lin > a_thr:
            self._last_accel_trip = s.t
            if s.t - self._last_gyro_trip <= C.SWING_COINCIDENCE_S:
                self._coincided = True

        # Report the swing at the strike, not when the follow-through finally
        # settles: once acceleration is clearly past its peak and the strike
        # window is complete, everything the event needs is known.
        if not self._spent and self._coincided and s.t - self._peak_lin_t >= C.SWING_STRIKE_WINDOW_S[1] \
                and lin < C.SWING_PAST_PEAK_RATIO * self._peak_lin:
            if self.stroke is not None and self.stroke.drawing_back(self._peak_lin_t):
                # That burst was the backswing: forget it and wait for the forward stroke.
                self._peak_lin, self._peak_lin_t, self._peak_gyro = 0.0, s.t, 0.0
                self._coincided = False
                return None
            self._spent = True
            ev = self._event()
            if ev is not None:
                return ev

        quiet = gyr < g_thr * C.SWING_HYSTERESIS and lin < a_thr * C.SWING_HYSTERESIS
        if quiet:
            self._quiet_since = self._quiet_since if self._quiet_since is not None else s.t
        else:
            self._quiet_since = None
        ended = (self._quiet_since is not None and s.t - self._quiet_since >= C.SWING_END_QUIET_S) \
            or s.t - self._start > C.SWING_MAX_DURATION_S
        if not ended:
            return None
        return self._finish()

    def _finish(self) -> Optional[SwingEvent]:
        """The swing has settled. Usually it was already reported at the strike."""
        self._active = False
        if self._spent or not self._coincided:
            return None
        if self.stroke is not None and self.stroke.drawing_back(self._peak_lin_t):
            return None
        return self._event()

    def _event(self) -> Optional[SwingEvent]:
        t_peak = self._peak_lin_t
        if t_peak - self._last_event_t < C.SWING_COOLDOWN_S:
            return None
        if t_peak - self._last_event_t < C.SWING_RECOVERY_S and \
                self._peak_lin < C.SWING_RECOVERY_RATIO * self._last_event_g:
            return None     # the weaker motion back to ready after a swing
        self._last_event_t = t_peak
        self._last_event_g = self._peak_lin
        cal = self.cal
        span = max(1e-6, cal.strength_max_g - cal.strength_min_g)
        strength = max(0.0, min(1.0, (self._peak_lin - cal.strength_min_g) / span))
        feats = swing_features(self._window, t_peak, cal, self.gravity)
        models = cal.models or {}
        use = lambda name: models.get(name) if models.get(name, {}).get("use") else None  # noqa: E731
        stroke = None
        if use("stroke"):
            stroke = "forehand" if swing_model.probability(use("stroke"), feats) >= 0.5 else "backhand"
        elif cal.stroke_terms:
            stroke = "forehand" if combine_terms(cal.stroke_terms, feats) >= 0 else "backhand"
        top_terms = cal.topspin_terms_backhand if (stroke == "backhand" and cal.topspin_terms_backhand) \
            else cal.topspin_terms
        top_model = use("topspin_bh") if stroke == "backhand" else use("topspin_fh")
        topspin = swing_model.signed(top_model, feats) if top_model else combine_terms(top_terms, feats)
        side_model = use("sidespin")
        sidespin = swing_model.signed(side_model, feats) if side_model else combine_terms(cal.sidespin_terms, feats)
        return SwingEvent(
            t_peak=t_peak,
            strength=strength,
            return_speed=strength_to_speed(strength),
            topspin=topspin,
            sidespin=sidespin,
            peak_accel_g=self._peak_lin,
            peak_gyro_dps=self._peak_gyro,
            features=feats,
            raw=list(self._window),
            imu_stroke=stroke,
        )


@dataclass
class TossEvent:
    t_peak: float
    strength: float               # 0..1
    height: float                 # m the ball is tossed above the hand


def toss_height(strength: float) -> float:
    s = max(0.0, min(1.0, strength))
    return C.TOSS_HEIGHT_MIN + (C.TOSS_HEIGHT_MAX - C.TOSS_HEIGHT_MIN) * s


class TossDetector:
    """Detects the serve toss: an upward jolt with little rotation. Shares the
    swing detector's gravity estimate and calibration; the thresholds come from
    the calibration (tools/calibrate_swing.py's toss step) or config.py.

    Every jolt it sees -- accepted or not -- is kept in `last` (peak upward g,
    max rotation, how "straight up" it was, and the verdict), so the game can
    tell the player why a flick didn't count. With record=True all of them are
    also appended to `candidates` (used by the calibration tool)."""

    def __init__(self, swing: "SwingDetector", record: bool = False):
        self.swing = swing
        self.record = record
        self.candidates: List[dict] = []
        self.last: Optional[dict] = None
        self._active = False
        self._peak = 0.0
        self._peak_t = 0.0
        self._max_gyro = 0.0
        self._up_frac = 0.0
        self._last_t = -math.inf

    def push(self, s: ImuSample) -> Optional[TossEvent]:
        cal = self.swing.cal
        thr = cal.toss_accel_g
        g = self.swing.gravity
        gmag = math.sqrt(sum(x * x for x in g)) or 1.0
        up = [x / gmag for x in g]               # at rest the accelerometer reads +1 g "up"
        a = accel_g(s, cal)
        lin = [a[i] - g[i] for i in range(3)]
        up_comp = sum(lin[i] * up[i] for i in range(3))
        lin_mag = math.sqrt(sum(x * x for x in lin)) or 1e-9
        gyr = gyro_dps(s, cal)
        if not self._active:
            if up_comp > thr * 0.5:
                self._active = True
                self._peak, self._peak_t, self._max_gyro, self._up_frac = up_comp, s.t, gyr, up_comp / lin_mag
            else:
                return None
        self._max_gyro = max(self._max_gyro, gyr)
        if up_comp > self._peak:
            self._peak, self._peak_t, self._up_frac = up_comp, s.t, up_comp / lin_mag
        if up_comp > thr * 0.3 and s.t - self._peak_t < 0.3:
            return None
        # Jolt over: was it a clean upward toss?
        self._active = False
        cand = {"t": self._peak_t, "peak_g": round(self._peak, 2), "gyro_dps": round(self._max_gyro, 1),
                "up_frac": round(self._up_frac, 2)}
        if self._peak < thr:
            cand["result"] = f"too gentle ({self._peak:.1f} g, needs {thr:.1f})"
        elif self._max_gyro > cal.toss_max_gyro_dps:
            cand["result"] = f"too much rotation ({self._max_gyro:.0f} deg/s, max {cal.toss_max_gyro_dps:.0f})"
        elif self._up_frac < cal.toss_up_fraction:
            cand["result"] = f"not straight up ({self._up_frac:.0%} upward, needs {cal.toss_up_fraction:.0%})"
        elif self._peak_t - self._last_t < C.TOSS_COOLDOWN_S:
            cand["result"] = "too soon after the last toss"
        else:
            cand["result"] = "toss"
        self.last = cand
        if self.record:
            self.candidates.append(cand)
        if cand["result"] != "toss":
            return None
        self._last_t = self._peak_t
        lo = cal.toss_peak_min_g if cal.toss_peak_min_g is not None else thr
        hi = cal.toss_peak_max_g if cal.toss_peak_max_g is not None else 3.0 * thr
        strength = max(0.0, min(1.0, (self._peak - lo) / max(1e-6, hi - lo)))
        return TossEvent(self._peak_t, strength, toss_height(strength))


def detect_all(samples: Sequence[ImuSample], cal: Optional[Calibration] = None) -> List[SwingEvent]:
    det = SwingDetector(cal)
    return [ev for ev in (det.push(s) for s in samples) if ev is not None]


def load_calibration(path: Optional[str]) -> Calibration:
    if path and os.path.exists(path):
        return Calibration.load(path)
    return Calibration()
