"""Simulated board for laptop dev and tests. The base class's dead-reckoned
pose is the simulator's ground truth, plus a little IMU noise."""
import random
import time

from hal import Hal


class SimHal(Hal):
    kind = "sim"
    imu_label = "sim"

    def __init__(self, latency_ms=8.0):
        super().__init__()
        self.connected = True
        self.latency_ms = latency_ms
        self.tripped = False
        self.drive_log = []            # (t, left, right) for tests

    def _drive(self, left, right):
        self.drive_log.append((time.time(), left, right))
        time.sleep(self.latency_ms / 1000 * random.uniform(0.5, 1.5))

    def _heartbeat(self):
        t, self.tripped = self.tripped, False
        return t

    def imu(self):
        r = super().imu()
        r["pitch"] = round(random.gauss(0, 0.3), 1)
        r["roll"] = round(random.gauss(0, 0.3), 1)
        return r
