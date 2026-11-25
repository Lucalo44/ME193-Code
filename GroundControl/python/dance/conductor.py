"""Conductor: one beat clock, N performers.

The LLM only writes a short score. The compiler turns it into timelines, and
the conductor plays them: every performer gets the same wall-clock start time
and runs its own scheduler (local thread for the car, the LEGO service's own
thread for the LEGO bot). Afterwards it measures how in sync they were.
"""
import statistics
import threading
import time

from dance.compiler import compile_score

PREP_S = 0.6        # time between "go" and the first count-in beat (lets every performer arm)


class Conductor:
    def __init__(self, performers: dict, emit=None, status=None):
        self.performers = performers          # name -> Performer
        self.emit = emit or (lambda *a, **k: None)
        self.status = status
        self._lock = threading.Lock()
        self._cancel = None
        self.last = None

    def stop_all(self):
        if self._cancel:
            self._cancel.set()
        for p in self.performers.values():
            try:
                p.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[conductor] stop {p.name}: {e!r}", flush=True)

    def perform(self, score, cancel=None, build=None):
        cancel = cancel or threading.Event()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("already dancing")
        try:
            self._cancel = cancel
            comp = compile_score(score, build=build)
            car = self.performers.get("car")
            if car is not None and hasattr(car, "calibrate"):
                car.calibrate()
            t_start = time.time() + PREP_S
            self.emit("dance.timeline", {**comp, "t_start": t_start, "server_now": time.time(),
                                         "performers": {n: p.state() for n, p in self.performers.items()}})
            reports = {}
            threads = []

            def on_kf(name, rep):
                err = None
                if rep.get("t_actual") is not None:
                    err = (rep["t_actual"] - rep["t_cmd"]) * 1000
                self.emit("dance.kf", {"performer": name, "i": rep["i"], "beat": rep["beat"],
                                       "action": rep.get("action"), "err_ms": None if err is None else round(err, 1),
                                       "motion": rep.get("motion", False)})

            def run(name, perf):
                try:
                    reports[name] = perf.play(comp["timelines"].get(name, []), t_start, cancel, on_kf)
                except Exception as e:  # noqa: BLE001
                    print(f"[conductor] {name} failed: {e!r}", flush=True)
                    reports[name] = []

            for name, perf in self.performers.items():
                th = threading.Thread(target=run, args=(name, perf), daemon=True, name=f"perf-{name}")
                th.start()
                threads.append(th)
            deadline = t_start + comp["seconds"] + 5
            for th in threads:
                th.join(max(0.0, deadline - time.time()))
            if cancel.is_set():
                self.stop_all()
            result = self.summarize(comp, reports, completed=not cancel.is_set())
            self.last = result
            self.emit("dance.done", result)
            return result
        finally:
            self._cancel = None
            self._lock.release()

    @staticmethod
    def summarize(comp, reports, completed=True):
        per = {}
        all_errs = []
        by_beat = {}
        for name, reps in reports.items():
            errs = [(r["t_actual"] - r["t_cmd"]) * 1000 for r in reps
                    if r.get("motion") and r.get("t_actual") is not None]
            all_errs += errs
            per[name] = {"n": len(errs),
                         "mean_ms": round(statistics.mean(errs), 1) if errs else None,
                         "max_abs_ms": round(max(abs(e) for e in errs), 1) if errs else None}
            for r in reps:
                if r.get("motion") and r.get("t_actual") is not None:
                    by_beat.setdefault(r["beat"], {})[name] = r["t_actual"]
        gaps = [abs(v["car"] - v["lego"]) * 1000 for v in by_beat.values() if "car" in v and "lego" in v]
        n_expected = sum(len(t) for t in comp["timelines"].values())
        n_done = sum(len(r) for r in reports.values())
        return {
            "completed": completed and n_done >= n_expected,
            "keyframes": n_done, "expected": n_expected,
            "bpm": comp["bpm"], "beats": comp["beats"], "seconds": comp["seconds"],
            "max_sync_error_ms": round(max(abs(e) for e in all_errs), 1) if all_errs else None,
            "mean_sync_error_ms": round(statistics.mean(abs(e) for e in all_errs), 1) if all_errs else None,
            "max_gap_ms": round(max(gaps), 1) if gaps else None,
            "performers": per,
        }
