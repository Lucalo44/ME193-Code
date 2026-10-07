"""
hardware/replay_paddle.py -- play a recorded IMU log back as if the Double Motor
were connected (python main.py --replay imu_logs/calib_hard.csv).

Every sample goes through the same path as the real paddle -- swing and toss
detection, orientation, the live stroke -- in real time, looping. Handy for
working on the game without the hardware, and for showing exactly what the
game makes of a recorded session.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from hardware.paddle import PaddleBase
from hardware.swing import Calibration, load_csv, retime


class ReplayPaddle(PaddleBase):
    kind = "replay"

    def __init__(self, calibration: Calibration, path: str, loop: bool = True):
        super().__init__(calibration)
        self.samples = retime(load_csv(path))
        self.loop = loop
        self.status = "replay"
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="replay").start()

    def _run(self) -> None:
        first = True
        while not self._stop.is_set():
            t0_log, t0 = self.samples[0].t, time.monotonic()
            for s in self.samples:
                delay = (s.t - t0_log) - (time.monotonic() - t0)
                if delay > 0:
                    time.sleep(delay)
                if self._stop.is_set():
                    return
                self._ingest(replace(s, t=time.monotonic()))
                if first and s.t - t0_log > 0.3:
                    self.zero()            # like the real paddle: zero on the opening still moment
                    first = False
            if not self.loop:
                return

    def shutdown(self) -> None:
        self._stop.set()
