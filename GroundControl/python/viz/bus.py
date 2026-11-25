"""The single place viz events leave Python.

Non-blocking: emit() drops into a bounded queue and returns; a sender thread
forwards to the WebUI (socket.io). Under load it drops the oldest high-rate
events and never back-pressures control code. It also keeps a snapshot of the
current/last job so a page that (re)connects mid-job can rebuild its view.
"""
import queue
import threading
import time
from collections import deque

import config

MIN_INTERVAL = {"imu": 0.05, "audio.mel": 0.03, "kws.probs": 0.1, "matrix": 0.03, "drive": 0.05}
JOB_EVENTS = {"job.new", "llm.prompt", "llm.token", "llm.retry", "llm.done", "plan.step", "job.done",
              "dance.timeline", "dance.kf", "dance.done", "mqtt.out"}


class VizBus:
    def __init__(self, ui=None):
        self.ui = ui
        self.q = queue.Queue(maxsize=config.VIZ_QUEUE)
        self._last = {}
        self._lock = threading.Lock()
        self.job = {}                  # current job snapshot for get_state
        self.history = deque(maxlen=8)  # finished job summaries (latency Gantt)
        self.latest = {}               # latest of non-job events (status, matrix, ...)
        self.mqtt_log = deque(maxlen=30)
        threading.Thread(target=self._run, daemon=True, name="viz").start()

    def emit(self, event, payload):
        now = time.time()
        mi = MIN_INTERVAL.get(event)
        if mi and now - self._last.get(event, 0) < mi:
            return
        self._last[event] = now
        self._remember(event, payload)
        if self.ui is None:
            return
        try:
            self.q.put_nowait((event, payload))
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait((event, payload))
            except queue.Empty:
                pass

    def _remember(self, event, p):
        with self._lock:
            if event == "job.new":
                self.job = {"job": p, "prompt": None, "tokens": [], "done": None, "steps": {},
                            "result": None, "dance": None, "dance_kf": [], "dance_done": None}
            elif event == "llm.prompt":
                self.job["prompt"] = p
            elif event == "llm.token":
                if p.get("attempt", 0) == 0 or not self.job.get("tokens"):
                    self.job.setdefault("tokens", []).append(p)
            elif event == "llm.done":
                self.job["done"] = p
            elif event == "plan.step":
                self.job.setdefault("steps", {})[str(p["i"])] = p
            elif event == "job.done":
                self.job["result"] = p
                self.history.append(p)
            elif event == "dance.timeline":
                self.job["dance"] = p
                self.job["dance_kf"] = []
            elif event == "dance.kf":
                self.job.setdefault("dance_kf", []).append(p)
            elif event == "dance.done":
                self.job["dance_done"] = p
            elif event == "mqtt.out":
                self.mqtt_log.append(p)
            if event not in JOB_EVENTS:
                self.latest[event] = p

    def snapshot(self):
        with self._lock:
            return {"job": self.job, "history": list(self.history), "latest": dict(self.latest),
                    "mqtt_log": list(self.mqtt_log), "server_now": time.time()}

    def _run(self):
        while True:
            event, payload = self.q.get()
            try:
                self.ui.send_message(event, payload)
            except Exception as e:  # noqa: BLE001
                print(f"[viz] send failed: {e!r}", flush=True)
