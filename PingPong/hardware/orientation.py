"""
hardware/orientation.py -- 3D orientation of the paddle for the on-screen paddle.

Two sources, chosen by ORIENTATION_SOURCE in config.py:

  * "onboard" (default): the Double Motor's own fused yaw/pitch/roll. Measured
    on our motor (imu_logs, Oct 7) these are aerospace Z-Y-X Euler angles in
    decidegrees -- yaw about the motor's z, then pitch about y, then roll about
    x -- which matched the gyro ~3x better than any other convention. Turned
    into a quaternion (never used as raw Euler angles on screen, so there is no
    gimbal flipping), it stayed within ~1 deg of gravity at every still moment,
    including right after sharp serve flicks.
  * "fusion": our own gyro + accelerometer filter below. It drifts on sharp
    motions (the gyro integration lost 40-50 deg over a set of serve flicks),
    so it is kept only as a fallback.

Either way the result is relative to the zeroed ready pose and mapped onto the
browser's scene frame with scene_mapping(). The fusion filter details:

  * OrientationFilter integrates the gyro (Mahony-style) and uses the
    accelerometer to stop "up" from drifting while the paddle isn't
    accelerating. Its frame is the motor's frame at the moment of zero() --
    the ready pose is the identity. With no magnetometer, heading (yaw) slowly
    drifts, so it is gently re-centered whenever the paddle is held still.
  * scene_mapping() turns that into the browser's scene frame (x right, y up,
    z toward the camera), using two directions measured in the motor frame:
    "up" (gravity at zero()) and the axis the paddle turns about when its face
    closes (tools/calibrate_swing.py measures it; a guess is used otherwise).

Quaternions are (w, x, y, z) tuples of floats.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

Quat = Tuple[float, float, float, float]
Vec = Tuple[float, float, float]

IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Small vector / quaternion helpers
# --------------------------------------------------------------------------

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Sequence[float]) -> Vec:
    n = norm(a)
    return (0.0, 0.0, 0.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qconj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def qnormalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def qaxis(axis: Sequence[float], angle: float) -> Quat:
    ax = unit(axis)
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), ax[0] * s, ax[1] * s, ax[2] * s)


def qrotate(q: Quat, v: Sequence[float]) -> Vec:
    w, x, y, z = qmul(qmul(q, (0.0, v[0], v[1], v[2])), qconj(q))
    return (x, y, z)


def qfrom_matrix(rows: Sequence[Sequence[float]]) -> Quat:
    """Quaternion of a rotation matrix given as rows."""
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = rows
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    return qnormalize(q)


def euler_yxz_deg(q: Quat) -> dict:
    """Scene-frame Euler angles (three.js 'YXZ' order) in degrees:
    pitch about x (negative = face closed), yaw about y, roll = -rotation about z."""
    w, x, y, z = q
    m02 = 2 * (x * z + w * y)
    m12 = 2 * (y * z - w * x)
    m22 = 1 - 2 * (x * x + y * y)
    m10 = 2 * (x * y + w * z)
    m11 = 1 - 2 * (x * x + z * z)
    pitch = math.asin(max(-1.0, min(1.0, -m12)))
    if abs(m12) < 0.9999:
        yaw = math.atan2(m02, m22)
        rz = math.atan2(m10, m11)
    else:
        yaw, rz = 0.0, 0.0
    return {"pitch": math.degrees(pitch), "yaw": math.degrees(yaw), "roll": -math.degrees(rz)}


# --------------------------------------------------------------------------
# Motor frame -> scene frame
# --------------------------------------------------------------------------

def scene_mapping(up: Sequence[float], close_axis: Optional[Sequence[float]]) -> Quat:
    """Rotation taking motor-frame (at zero) coordinates to scene coordinates.

    up:         gravity direction measured at zero() (accelerometer at rest reads +1 g up)
    close_axis: rotation axis (right-hand rule) of closing the paddle face, in
                the motor frame. Closing the face tips it toward the table, which
                is a rotation about scene -x. If unknown, the motor axis most
                perpendicular to "up" is used as a guess.
    """
    y = unit(up)
    if close_axis is None or norm(close_axis) < 1e-6:
        axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        close_axis = min(axes, key=lambda a: abs(dot(a, y)))
    c = unit(close_axis)
    x = unit(tuple(-(c[i] - dot(c, y) * y[i]) for i in range(3)))
    z = cross(x, y)
    return qfrom_matrix([x, y, z])


# --------------------------------------------------------------------------
# Gyro + accelerometer fusion
# --------------------------------------------------------------------------

class OrientationFilter:
    KP = 1.5               # rad/s per unit tilt error: how fast "up" is corrected from the accelerometer
    ACCEL_TRUST_G = 0.15   # only trust the accelerometer when |a| is within this of 1 g
    STILL_DPS = 12.0       # below this the paddle counts as "held still"
    STILL_S = 0.8          # ... for this long before heading is re-centered
    RECENTER_PER_S = 0.4   # fraction of heading error removed per second while still

    def __init__(self):
        self.q: Quat = IDENTITY
        self.up: Vec = (0.0, 0.0, 1.0)
        self.ref: Vec = (1.0, 0.0, 0.0)     # a horizontal reference direction for heading
        self._last_t: Optional[float] = None
        self._still_since: Optional[float] = None

    def reset(self, up: Sequence[float]) -> None:
        """Make the current pose the identity (call at zero(), paddle still)."""
        self.q = IDENTITY
        self.up = unit(up)
        axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        a = min(axes, key=lambda v: abs(dot(v, self.up)))
        self.ref = unit(tuple(a[i] - dot(a, self.up) * self.up[i] for i in range(3)))
        self._last_t = None
        self._still_since = None

    def update(self, t: float, gyro_dps: Sequence[float], accel_g: Sequence[float]) -> Quat:
        """Advance with one sample: gyro in deg/s (bias removed), accel in g, both motor frame."""
        if self._last_t is None:
            self._last_t = t
            return self.q
        dt = min(0.05, max(0.0, t - self._last_t))
        self._last_t = t
        w = [math.radians(g) for g in gyro_dps]
        # Pull the estimated "up" toward the measured one while not accelerating.
        a_mag = norm(accel_g)
        if abs(a_mag - 1.0) < self.ACCEL_TRUST_G:
            v_pred = qrotate(qconj(self.q), self.up)          # world up seen from the motor
            e = cross(unit(accel_g), v_pred)
            w = [w[i] + self.KP * e[i] for i in range(3)]
        rate = norm(w)
        if rate > 1e-9:
            self.q = qnormalize(qmul(self.q, qaxis(w, rate * dt)))
        # Re-center heading while held still (there's no compass to fix yaw drift).
        if norm(gyro_dps) < self.STILL_DPS:
            self._still_since = self._still_since if self._still_since is not None else t
            if t - self._still_since > self.STILL_S:
                self.q = qnormalize(qmul(qaxis(self.up, -self.heading() * self.RECENTER_PER_S * dt), self.q))
        else:
            self._still_since = None
        return self.q

    def heading(self) -> float:
        """Signed rotation (rad) about world up of the current pose."""
        r = qrotate(self.q, self.ref)
        r = tuple(r[i] - dot(r, self.up) * self.up[i] for i in range(3))
        if norm(r) < 1e-6:
            return 0.0
        r = unit(r)
        return math.atan2(dot(cross(self.ref, r), self.up), dot(self.ref, r))


def to_scene(q_motor: Quat, mapping: Quat) -> Quat:
    """Express a motor-frame rotation in the scene frame."""
    return qnormalize(qmul(qmul(mapping, q_motor), qconj(mapping)))


# --------------------------------------------------------------------------
# The motor's own orientation
# --------------------------------------------------------------------------

def onboard_quat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Quat:
    """Motor body -> motor world, from its Z-Y-X (yaw, pitch, roll) Euler angles."""
    return qmul(qmul(qaxis((0.0, 0.0, 1.0), math.radians(yaw_deg)), qaxis((0.0, 1.0, 0.0), math.radians(pitch_deg))),
                qaxis((1.0, 0.0, 0.0), math.radians(roll_deg)))


class OnboardOrientation:
    """Orientation relative to the zeroed pose, from the motor's fused angles."""

    def __init__(self):
        self.q0_inv: Quat = IDENTITY
        self.q: Quat = IDENTITY            # current body -> zero-pose body (same meaning as OrientationFilter.q)
        self._last: Optional[Quat] = None

    def update(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> Quat:
        self._last = onboard_quat(yaw_deg, pitch_deg, roll_deg)
        self.q = qnormalize(qmul(self.q0_inv, self._last))
        return self.q

    def reset(self) -> None:
        """Make the latest pose the zero (ready) pose."""
        if self._last is not None:
            self.q0_inv = qconj(self._last)
            self.q = IDENTITY
