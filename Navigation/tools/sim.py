"""Run the navigator on the laptop with a recorded walk standing in for the ankle IMU.

Serves the same two pages as the board, so the phone UI and the dashboard can be tried (and
demoed) with no hardware:
    python3 tools/sim.py                          # newest real walk, real time
    python3 tools/sim.py --test                   # a synthetic walk from data/test_walks
    python3 tools/sim.py --walk data/walks/luca/<id> --speed 2
then open  http://localhost:8080/  (phone view)  and  http://localhost:8080/dash  (dashboard).
Pressing Navigate / Record / DAgger restarts the walk from the beginning. Sim recordings are
saved to data/sim_walks/, never into the real dataset.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from nav.server import NavServer  # noqa: E402
from nav.session import Session  # noqa: E402
from nav.walks import load_walk, walk_dirs  # noqa: E402


class SimSession(Session):
    """A Session whose sample source rewinds whenever a new mode starts."""

    def __init__(self, imu, speed, **kw):
        super().__init__(**kw)
        self.imu, self.speed, self.pos = imu, speed, 0

    def start(self, mode, person):
        with self.lock:
            self.pos = 0
            super().start(mode, person)

    def run(self):
        chunk = 5
        while True:
            t0 = time.monotonic()
            with self.lock:
                if self.pos >= len(self.imu) and self.mode == "idle":
                    self.pos = 0                          # loop while idle so the dashboard stays live
                rows = self.imu[self.pos:self.pos + chunk]  # empty once a walk has finished
                self.pos += len(rows)
            if len(rows):
                self.feed(rows)
            time.sleep(max(0.0, chunk / C.FS / self.speed - (time.monotonic() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", help="walk folder to replay")
    ap.add_argument("--test", action="store_true", help="use a synthetic walk (data/test_walks)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--models", default=str(C.MODEL_DIR), help="model folder (policy.npz, stairs.json, ...)")
    a = ap.parse_args()

    if a.walk:
        path = Path(a.walk)
    else:
        dirs = walk_dirs(C.TEST_DIR if a.test else C.DATA_DIR)
        if not dirs:
            raise SystemExit("No walks to replay. Record one, or run tools/make_test_walks.py and use --test.")
        path = dirs[-1]
    w = load_walk(path)
    print(f"replaying {path} ({len(w['imu']) / C.FS:.0f} s) at {a.speed}x")
    s = SimSession(w["imu"], a.speed, walk_root=C.HERE / "data" / "sim_walks", model_dir=Path(a.models))
    print("navigator:", "loaded" if s.nav.policy.loaded else "not trained (navigate mode will say so)")
    NavServer(s, a.port)
    threading.Thread(target=s.run, daemon=True).start()
    print(f"open http://localhost:{a.port}/  and  http://localhost:{a.port}/dash   (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.stop()


if __name__ == "__main__":
    main()
