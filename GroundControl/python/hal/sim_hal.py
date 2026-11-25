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

    # A simulated bench: something driving A0, a Modulino Movement on the Qwiic bus.
    def _probe_pins(self):
        return "D2:f,D3:f,D4:f,D7:f,D8:f,D11:f,D12:f,D13:f,A0:H,A1:f,A2:f,A3:f,A4:f,A5:f,"

    def _i2c_scan(self):
        return "Wire:;Wire1:6a;Wire2:"

    def imu(self):
        r = super().imu()
        r["pitch"] = round(random.gauss(0, 0.3), 1)
        r["roll"] = round(random.gauss(0, 0.3), 1)
        return r
