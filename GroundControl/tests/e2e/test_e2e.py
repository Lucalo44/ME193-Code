"""End-to-end (Part 2, A2): commands go in over a real MQTT broker and come out as matrix
frames, MQTT replies and dance timelines, through the whole app (sim board, fake LLM).

Needs mosquitto (brew install mosquitto, or the eclipse-mosquitto Docker image); skipped
without it. Each test run starts its own broker on a free port with a websockets listener,
like test.mosquitto.org's.
"""
import json
import shutil
import socket
import subprocess
import threading
import time

import pytest

import config

MOSQUITTO = shutil.which("mosquitto") or "/opt/homebrew/sbin/mosquitto"
pytestmark = pytest.mark.skipif(not shutil.which(MOSQUITTO), reason="mosquitto not installed")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def broker(tmp_path):
    port = free_port()
    conf = tmp_path / "mosquitto.conf"
    conf.write_text(f"listener {port} 127.0.0.1\nprotocol websockets\nallow_anonymous true\n")
    p = subprocess.Popen([MOSQUITTO, "-c", str(conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    yield port
    p.terminate()
    p.wait(5)


class Client:
    """A test client on the same broker: sends commands, records everything on ME193/q/luca/# and luca."""

    def __init__(self, port):
        import paho.mqtt.client as mqtt
        self.msgs, self.ready = [], threading.Event()
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="websockets")
        c.ws_set_options(path=config.BROKER_WS_PATH)
        c.on_connect = lambda c, u, f, rc, p=None: (c.subscribe([(f"{config.TOPIC_BASE}/#", 0), (config.NAME, 0)]),
                                                     self.ready.set())
        c.on_message = lambda c, u, m: self.msgs.append((m.topic, m.payload.decode(errors="replace"), time.time()))
        c.connect("127.0.0.1", port)
        c.loop_start()
        self.c = c
        assert self.ready.wait(5)

    def send(self, payload):
        t = time.time()
        self.c.publish(config.TOPIC_CMD, payload if isinstance(payload, str) else json.dumps(payload))
        return t

    def wait(self, pred, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for m in list(self.msgs):
                if pred(m):
                    return m
            time.sleep(0.02)
        return None

    def close(self):
        self.c.loop_stop()
        self.c.disconnect()


@pytest.fixture
def stack(broker, tmp_path, monkeypatch):
    from command_log import CommandLog
    from core import GroundControl
    from hal.sim_hal import SimHal
    from memory_store import Memory
    from planner.fake_llm import FakeLLM
    monkeypatch.setattr(config, "BROKER_HOST", "127.0.0.1")
    monkeypatch.setattr(config, "BROKER_PORT", broker)
    monkeypatch.setattr(config, "MIN_JOB_INTERVAL_S", 0)
    g = GroundControl(SimHal(latency_ms=1), ui=None, llm=FakeLLM(delay_s=0.001), lego=False, mqtt=True)
    g.memory = Memory(tmp_path / "memory.json")
    g.command_log = CommandLog(tmp_path / "commands.jsonl")
    g.start(warm=False)
    assert _until(lambda: g.status.get("mqtt"), 5)
    cl = Client(broker)
    yield g, cl
    cl.close()
    g.mqtt.stop()


def _until(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


DANCE = {"steps": [{"skill": "dance", "args": {"bpm": 60, "sections": [
    {"bars": 2, "mode": "mirror", "car": "shimmy", "lego": "wave"}], "where": "matrix"}}]}


def test_assignment_example_text_on_the_matrix_and_reply_on_luca(stack):
    g, cl = stack
    cl.send("Type Hi on the LED matrix and then send back on topic luca 'I am done'")
    assert cl.wait(lambda m: m[0] == config.NAME and m[1] == "I am done")
    assert g.matrix.content.get("text") == "Hi"
    assert any(v for v in g.hal.frame)                       # something is lit on the (sim) matrix


def test_icon_command_draws_the_exact_icon(stack):
    from skills.font import icon_frame
    g, cl = stack
    cl.send("show a heart")
    assert _until(lambda: g.matrix.content.get("name") == "heart")
    assert _until(lambda: g.hal.frame == icon_frame("heart"))


def test_status_answers_in_under_a_second_mid_plan(stack):
    g, cl = stack
    cl.send({"plan": DANCE, "text": "loop it"})
    assert _until(lambda: g.status.get("state") == "DANCING", 10)
    t = cl.send("/status")
    m = cl.wait(lambda m: m[0] == config.TOPIC_REPLY and m[1].startswith("{") and m[2] >= t, 3)
    assert m and m[2] - t < 1.0 and json.loads(m[1])["state"] == "DANCING"


def test_stop_halts_a_running_dance(stack):
    g, cl = stack
    cl.send({"plan": DANCE, "text": "loop it"})
    assert _until(lambda: g.status.get("state") == "DANCING", 10)
    t = cl.send("stop dancing")
    assert cl.wait(lambda m: m[0] == config.TOPIC_REPLY and m[1] == "stopped" and m[2] >= t, 3)
    assert _until(lambda: g.queue.current is None, 3)


def test_dance_timeline_keyframe_counts(stack):
    from dance.compiler import compile_score
    from dance.figure import figure_timeline
    g, cl = stack
    plan = {"steps": [{"skill": "dance", "args": {"bpm": 160, "sections": [
        {"bars": 1, "mode": "unison", "car": "step", "lego": "clap"}], "where": "matrix"}}]}
    cl.send({"plan": plan, "text": "dance"})
    assert _until(lambda: (g.status.get("dance") or {}).get("completed"), 15)
    comp = compile_score({"bpm": 160, "dancers": 1, "sections": plan["steps"][0]["args"]["sections"]})
    d = g.status.get("dance")
    assert d["expected"] == d["keyframes"] == len(figure_timeline(comp))
