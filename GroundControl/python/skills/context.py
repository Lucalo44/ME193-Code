"""What a skill gets to touch while it runs."""
import threading
import time
from dataclasses import dataclass, field


class SkillError(RuntimeError):
    pass


class Cancelled(SkillError):
    pass


@dataclass
class Ctx:
    hal: object
    matrix: object
    publish: callable                     # (topic, message) -> None
    status: object
    conductor: object = None
    imu_source: callable = None           # () -> dict | None
    reply_to: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    step_timeout: threading.Event = field(default_factory=threading.Event)
    published: list = field(default_factory=list)
    answers: list = field(default_factory=list)

    def cancelled(self):
        return self.cancel.is_set() or self.step_timeout.is_set()

    def sleep(self, seconds):
        """Interruptible sleep. Raises Cancelled on e-stop or step timeout."""
        end = time.monotonic() + seconds
        while True:
            if self.cancelled():
                raise Cancelled("cancelled" if self.cancel.is_set() else "step timed out")
            rem = end - time.monotonic()
            if rem <= 0:
                return
            self.cancel.wait(min(rem, 0.02))

    def imu(self):
        if self.imu_source:
            r = self.imu_source()
            if r:
                return r
        return self.hal.imu()
