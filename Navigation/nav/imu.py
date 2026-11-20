"""IMU input: parse the samples the sketch queues (Bridge "get_imu") and track heading.

The device swings in a hand or pocket at any angle, so there is no fixed yaw axis. Projecting
the gyro onto a low-passed gravity vector doesn't work while walking: the sway makes the
"up" estimate wobble, and that wobble times the swing rate becomes a fake steady rotation
(hundreds of degrees per minute on our walk data).

Instead this runs a small attitude filter (Mahony-style, no magnetometer): the gyro is
integrated into a full orientation quaternion, and the accelerometer corrects the tilt. Our
StepNet's MPU-6050 rides on the ankle, where the shank swings at 300-500 deg/s every stride, so
the accelerometer is only trusted during stance (foot planted: low rotation, ~1 g). That's the
zero-velocity-update trick used for foot-mounted IMUs; during the swing the gyro runs alone. Heading is the integral of the angular velocity about the
true world vertical. It still drifts slowly (there's no compass), but turns come out as clean
90-degree steps, which is what the navigator needs.
"""
import math
from collections import deque

import numpy as np


def parse_drain(text):
    """One get_imu() reply "ms,ax,ay,az,gx,gy,gz;..." -> (n, 7) array: t (s), accel (g), gyro (deg/s)."""
    out = []
    for r in (text or "").split(";"):
        p = r.split(",")
        if len(p) != 7:
            continue
        try:
            v = [float(x) for x in p]
        except ValueError:
            continue
        v[0] /= 1000.0
        out.append(v)
    return np.array(out, dtype=float).reshape(-1, 7)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rotate(q, v):
    """Rotate body vector v into the world frame with q (world <- body)."""
    w, x, y, z = q
    return np.array([
        (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - w * z) * v[1] + 2 * (x * z + w * y) * v[2],
        2 * (x * y + w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - w * x) * v[2],
        2 * (x * z - w * y) * v[0] + 2 * (y * z + w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2]])


def _tilt_quat(a):
    """Quaternion that rotates the measured gravity direction a onto world +z."""
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(a, z))
    if c < -0.9999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(a, z)
    q = np.array([1.0 + c, *axis])
    return q / np.linalg.norm(q)


class Heading:
    """Attitude filter. heading: deg, + = counter-clockwise = left turn."""

    KP_STANCE = 4.0               # tilt correction gain while the foot is planted (rad/s per unit error)
    STANCE_GYRO = 60.0            # deg/s: below this (and ~1 g) the foot counts as planted

    def __init__(self, fs, tau_s=None):
        self.dt = 1.0 / fs
        self.q = None
        self.heading = 0.0
        self.bias = np.zeros(3)   # gyro bias (deg/s, body axes), learned whenever the device is still
        self.recent = deque(maxlen=int(fs * 0.5))
        self.yaw_rate = 0.0
        self.up_body = np.array([0.0, 0.0, 1.0])

    def update(self, samples):
        """samples: (n, 6) accel (g) + gyro (deg/s). Returns the heading after the last sample."""
        for a, graw in zip(samples[:, :3], samples[:, 3:6]):
            na = float(np.linalg.norm(a))
            # still = the gyro is steady (not small: the raw MPU-6050 bias can be several deg/s)
            # (accel scale can be off by ~10% too, so "still" is steady accel + steady gyro)
            self.recent.append(np.concatenate([a, graw]))
            if len(self.recent) == self.recent.maxlen and abs(na - 1.0) < 0.2:
                r = np.array(self.recent)
                steady = r[:, 3:].std(0).max() < 1.5 and r[:, :3].std(0).max() < 0.02
                if steady and np.linalg.norm(r[:, 3:].mean(0)) < 25:   # a steady turn is not a bias
                    self.bias += 0.05 * (r[:, 3:].mean(0) - self.bias)
            gdeg = graw - self.bias
            if self.q is None:
                if na < 0.3:
                    continue
                self.q = _tilt_quat(a)
            w = np.radians(gdeg)
            # where the filter thinks "up" is, in body coordinates
            qc = self.q * np.array([1, -1, -1, -1])
            up = _rotate(qc, np.array([0.0, 0.0, 1.0]))
            self.up_body = up
            if np.linalg.norm(gdeg) < self.STANCE_GYRO and abs(na - 1.0) < 0.15:   # stance: fix tilt
                w = w + self.KP_STANCE * np.cross(a / na, up)
            self.q = self.q + 0.5 * _qmul(self.q, np.array([0.0, *w])) * self.dt
            self.q /= np.linalg.norm(self.q)
            self.yaw_rate = math.degrees(float(_rotate(self.q, np.radians(gdeg))[2]))   # about world vertical
            self.heading += self.yaw_rate * self.dt
        return self.heading

    def vertical(self, samples):
        """Acceleration along the filter's up direction, minus 1 g (stairs features)."""
        return samples[:, :3] @ self.up_body - 1.0
