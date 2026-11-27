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
def gc(tmp_path):
    g = GroundControl(SimHal(latency_ms=1), ui=None, llm=FakeLLM(delay_s=0.001), lego=False, mqtt=False)
    from memory_store import Memory
    g.memory = Memory(tmp_path / "memory.json")            # never write memory.json into the repo
    from command_log import CommandLog
    g.command_log = CommandLog(tmp_path / "commands.jsonl")
    sent = []
    g.mqtt.on_publish_hook = lambda t, p: sent.append((t, p))
    g.sent = sent
    g.start(warm=False)
    yield g
    g._stop.set()
