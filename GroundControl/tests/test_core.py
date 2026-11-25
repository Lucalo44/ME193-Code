import json
import threading
import time

import config
from executor import resolve
from jobs import Job, JobQueue, QueueFull
from mqtt_io import parse_payload, topic_allowed
from planner.planner import JsonFieldTracker, group_alternatives
from planner.prompt import system_prompt
from skills.registry import REGISTRY, plan_schema, validate_plan


def wait_for(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------- registry
def test_schema_has_think_first_and_all_skills():
    s = plan_schema(60)
    assert list(s["properties"]) == ["parts", "steps"]
    names = [b["properties"]["skill"]["const"] for b in s["properties"]["steps"]["items"]["anyOf"]]
    assert set(names) == set(REGISTRY) - {"read_imu"}       # read_imu: placeholders read on demand
    assert "parts" not in plan_schema(0)["properties"]


def test_repair_schema_pins_sequence():
    s = plan_schema(60, fixed=["type Hi", "tell luca done"])
    st = s["properties"]["steps"]
    assert st["minItems"] == st["maxItems"] == 2 and len(st["prefixItems"]) == 2
    assert s["properties"]["parts"] == {"const": ["type Hi", "tell luca done"]}


def test_args_are_clamped():
    steps = validate_plan({"steps": [{"skill": "drive", "args": {"speed": 900, "seconds": 99}},
                                     {"skill": "wait", "args": {"seconds": -3}}]})
    assert steps[0][1] == {"speed": 100, "seconds": 5.0}
    assert steps[1][1] == {"seconds": 0.0}


def test_unknown_skill_and_bad_enum_are_errors():
    steps = validate_plan({"steps": [{"skill": "rm_rf", "args": {}},
                                     {"skill": "matrix_icon", "args": {"name": "skull"}}]})
    assert steps[0][0] is None
    assert "skull" in str(steps[1][1])


# ---------------------------------------------------------------- MQTT input
def test_topic_whitelist():
    assert topic_allowed("luca")
    assert topic_allowed("ME193/q/luca/reply")
    assert not topic_allowed("ME193/q/luca/cmd")           # no command loops
    assert not topic_allowed("ME193/q/fred/cmd")           # no injecting into classmates
    assert not topic_allowed("ME193/#")
    assert not topic_allowed("fred")


def test_parse_payload_plain_and_json():
    assert parse_payload(b"type Hi")["text"] == "type Hi"
    d = parse_payload(json.dumps({"text": "hi", "reply_to": "luca", "source": "voice"}).encode())
    assert d == {"text": "hi", "reply_to": "luca", "id": None, "source": "voice", "plan": None}
    d = parse_payload(json.dumps({"text": "hi", "reply_to": "ME193/q/fred/cmd"}).encode())
    assert d["reply_to"] is None
    assert len(parse_payload(b"x" * 5000)["text"]) == config.MAX_CMD_CHARS


def test_queue_cap():
    q = JobQueue(cap=2)
    q.put(Job("a"))
    q.put(Job("b"))
    try:
        q.put(Job("c"))
        assert False
    except QueueFull:
        pass


# ---------------------------------------------------------------- executor
def test_placeholders():
    r = {"imu": {"yaw": 92.4321, "pitch": 1.0}}
    assert resolve("yaw is {imu.yaw} deg", r) == "yaw is 92.4 deg"
    assert resolve("{yaw}!", r) == "92.4!"
    assert resolve("{nope.x}", r) == "{nope.x}"


def test_fred_example_end_to_end(gc):
    ok, _ = gc.submit("Type Hi on the LED matrix and then send back on MQTT on topic luca message 'I am done'")
    assert ok
    assert wait_for(lambda: ("luca", "I am done") in gc.sent)
    assert gc.matrix.content == {"kind": "text", "text": "Hi"}


def test_info_request_returns_real_value(gc):
    gc.hal.yaw = 37.0
    gc.submit("what is your yaw? reply on luca")
    assert wait_for(lambda: any(t == "luca" and p.startswith("yaw is ") for t, p in gc.sent))
    msg = next(p for t, p in gc.sent if t == "luca")
    assert abs(float(msg.split()[2]) - 37.0) < 1.0


def test_done_reply_when_nothing_asked(gc):
    gc.submit("show a heart")
    assert wait_for(lambda: (config.TOPIC_REPLY, "done") in gc.sent)


def test_refusal(gc):
    gc.submit("make me a sandwich")
    assert wait_for(lambda: any(p.startswith("Sorry, I can't do that") for _, p in gc.sent))


def test_status_answers_while_planning(gc):
    gc.llm.delay_s = 0.05                     # slow "model"
    gc.submit("Type Hi and tell me on luca")
    assert wait_for(lambda: gc.status.get("state") == "PLANNING")
    t0 = time.time()
    gc.mqtt.handle({"text": "/status", "source": "mqtt"})
    dt = time.time() - t0
    st = json.loads(next(p for t, p in gc.sent if t == config.TOPIC_REPLY))
    assert st["state"] == "PLANNING" and dt < 0.2


def test_estop_preempts_drive(gc):
    job = Job("drive")
    ctx = gc.make_ctx(job)

    def run():
        try:
            REGISTRY["drive"].fn(ctx, speed=80, seconds=5)
        except Exception:
            pass
    th = threading.Thread(target=run)
    th.start()
    assert wait_for(lambda: gc.hal.left != 0)
    gc.queue.current = job
    t0 = time.time()
    gc.estop("test")
    th.join(1)
    assert not th.is_alive()
    assert gc.hal.left == 0 and gc.hal.right == 0
    assert time.time() - t0 < 0.3


# ---------------------------------------------------------------- planner bits
def test_json_tracker_labels_decisions():
    tr = JsonFieldTracker()
    pieces = ['{"', 'think', '":', ' "', 'x', '",', ' "', 'steps', '":', ' [{"', 'skill', '":', ' "',
              'matrix', '_text', '",', ' "', 'args', '":', ' {"', 'text', '":', ' "', 'Hi', '"}}]}']
    out = [tr.feed(p) for p in pieces]
    i = pieces.index("matrix")
    assert out[i] == ("skill", True)
    assert out[i + 1] == ("skill", False)
    assert out[pieces.index("Hi")] == ("text", True)


def test_declared_in_think():
    from planner.planner import declared_in_think
    assert declared_in_think("matrix_text, then mqtt_publish to luca") == ["matrix_text", "mqtt_publish"]
    assert declared_in_think("no skill for that") == []


def test_group_alternatives():
    g = group_alternatives("skill", [{"piece": "matrix", "p": 0.8}, {"piece": " mq", "p": 0.1}])
    assert g[0]["choice"].startswith("matrix_text|matrix_icon")
    assert g[1]["choice"] == "mqtt_publish"


def test_prompt_is_byte_identical():
    assert system_prompt(60) == system_prompt(60)
    assert "{imu.yaw}" in system_prompt(60)


def test_imu_placeholder_reads_on_demand(gc):
    gc.hal.yaw = 123.0
    gc.mqtt.handle({"text": "p", "source": "mqtt", "plan": {"steps": [
        {"skill": "mqtt_publish", "args": {"topic": "luca", "message": "heading {imu.yaw}"}}]}})
    assert wait_for(lambda: any(t == "luca" for t, _ in gc.sent))
    msg = next(p for t, p in gc.sent if t == "luca")
    assert abs(float(msg.split()[1]) - 123.0) < 1.0


def test_hand_written_plan_skips_llm(gc):
    plan = {"steps": [{"skill": "matrix_icon", "args": {"name": "star"}},
                      {"skill": "mqtt_publish", "args": {"topic": "ME193/q/fred/cmd", "message": "x"}},
                      {"skill": "mqtt_publish", "args": {"topic": "luca", "message": "ok"}}]}
    gc.mqtt.handle({"text": "p", "source": "mqtt", "plan": plan})
    assert wait_for(lambda: ("luca", "ok") in gc.sent)
    assert not any(t == "ME193/q/fred/cmd" for t, _ in gc.sent)     # whitelist still applies
    assert gc.matrix.content == {"kind": "icon", "name": "star"}


def test_general_question_is_answered(gc):
    gc.submit("what is 2+2?")
    assert wait_for(lambda: (config.TOPIC_REPLY, "2 + 2 is 4.") in gc.sent)
    assert gc.matrix.content == {"kind": "text", "text": "2 + 2 is 4."}
    assert not any(p == "done" for _, p in gc.sent)          # the answer IS the reply


def test_self_report_uses_real_probes(gc):
    import time as _t
    gc.self_model._host_cache = (_t.time(), {"model": "Arduino UnoQ", "hostname": "Luca", "soc": ["qcom,qrb2210"],
                                            "cpus": 4, "mem_total_mb": 3667, "os": "Debian GNU/Linux 13",
                                            "kernel": "7.0"})
    gc.submit("what board are you on and is anything plugged into your pins?")
    assert wait_for(lambda: len([p for t, p in gc.sent if t == config.TOPIC_REPLY]) >= 2)
    board, pins = [p for t, p in gc.sent if t == config.TOPIC_REPLY][:2]
    assert "Arduino UNO Q" in board and "Qualcomm QRB2210" in board and "3.6 GB" in board
    assert "D5/D6" in pins and "A0 (held high)" in pins and "0x6a on the Qwiic connector" in pins
