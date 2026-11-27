"""Jobs: one command from MQTT / voice / dashboard, its id, and its cancel flag."""
import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import config

_ids = itertools.count(1)


@dataclass
class Job:
    text: str
    source: str = "mqtt"                 # mqtt | voice | ui | reflex | eval
    reply_to: str = config.TOPIC_REPLY
    id: str = field(default_factory=lambda: f"j{next(_ids)}")
    created: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    plan: dict | None = None
    reply: str | None = None
    timings: dict = field(default_factory=dict)   # stage -> (start, end) epoch s
    raw: str | None = None                # the text as it arrived (STT transcript), before normalize
    changes: list = field(default_factory=list)   # what the normalizer changed: [(before, after)]

    def mark(self, stage, start, end=None):
        self.timings[stage] = (start, end if end is not None else time.time())


class QueueFull(Exception):
    pass


class JobQueue:
    """FIFO with a cap. One worker pops; anyone can cancel."""

    def __init__(self, cap=config.QUEUE_CAP):
        self.cap = cap
        self._q = deque()
        self._cv = threading.Condition()
        self.current: Job | None = None

    def put(self, job: Job):
        with self._cv:
            if len(self._q) >= self.cap:
                raise QueueFull(f"queue full ({self.cap})")
            self._q.append(job)
            self._cv.notify()

    def get(self, timeout=None) -> Job | None:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            if not self._q:
                return None
            self.current = self._q.popleft()
            return self.current

    def done(self):
        with self._cv:
            self.current = None

    def __len__(self):
        with self._cv:
            return len(self._q)

    def cancel_all(self):
        """E-stop: cancel the running job and drop everything waiting."""
        with self._cv:
            dropped = list(self._q)
            self._q.clear()
            cur = self.current
        for j in dropped:
            j.cancel.set()
        if cur:
            cur.cancel.set()
        return cur, dropped
