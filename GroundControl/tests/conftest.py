import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
os.environ.setdefault("GC_SIM", "1")

import pytest  # noqa: E402

from core import GroundControl  # noqa: E402
from hal.sim_hal import SimHal  # noqa: E402
from planner.fake_llm import FakeLLM  # noqa: E402


class RecordingMqtt:
    pass


@pytest.fixture
def gc():
    g = GroundControl(SimHal(latency_ms=1), ui=None, llm=FakeLLM(delay_s=0.001), lego=False, mqtt=False)
    sent = []
    g.mqtt.on_publish_hook = lambda t, p: sent.append((t, p))
    g.sent = sent
    g.start(warm=False)
    yield g
    g._stop.set()
