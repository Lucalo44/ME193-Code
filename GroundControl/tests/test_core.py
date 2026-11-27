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
    assert "D5, D6 left wheel" in pins and "A0 held high" in pins and "0x6a on Qwiic connector" in pins
    assert "open pins (nothing attached): 13 of the 14" in pins   # the composer gets the count, not just a list


def test_facts_follow_the_question_words(gc):
    f = gc.self_model.facts("devices", "how much disk space do you have left?")
    assert "disk free" in f                      # health facts pulled in by the word "disk"



def test_ft_prompt_matches_training_and_has_no_examples():
    from planner import prompt as P
    import config
    msgs = P.messages("dance", 60, "ft")
    assert [m["role"] for m in msgs] == ["system", "user"]              # no few-shot turns
    assert msgs[0]["content"] == P.system_prompt(60, "think")            # == gen_data.py's SYSTEM
    assert config.style_for(config.FT_MODEL) == "ft" and config.style_for("other") == "think"
    assert config.style_for("custom/gc-planner-v2-Q4_0") == "ft"



def test_loop_count_reads_the_users_words():
    from skills.dance_skill import loop_count
    assert loop_count("loop a stick figure dance on the matrix") == 0          # until STOP
    assert loop_count("dance a waltz") == 1
    assert loop_count("loop the dance 3 times") == 3
    assert loop_count("repeat it twice") == 2
    assert loop_count("spin twice then dance") == 1                          # "twice" belongs to the spin
    assert loop_count("loop a stick figure dance on the matrix 2 times") == 2   # count far from "loop"
    assert loop_count("loop a dance, then spin 3 times") == 0                 # the 3 belongs to the spin


def test_perform_looped_skips_count_in_after_first_pass():
    import threading
    from types import SimpleNamespace
    from skills.dance_skill import perform_looped
    calls = []

    class Cond:
        def perform(self, score, cancel=None, where="robots", count_in=None):
            calls.append(count_in)
            return {"seconds": 1.0, "beats": 16}
    ctx = SimpleNamespace(question="loop it 3 times", conductor=Cond(), cancel=threading.Event())
    r = perform_looped(ctx, {"bpm": 100, "sections": []}, "matrix")
    assert calls == [None, 0, 0] and r["loops"] == 3 and r["looped"]


def test_facts_keep_the_sentence_that_answers():
    from wiki import select_facts
    res = [{"title": "Tower", "extract": "The Tower is a skyscraper in a city. It was designed by someone famous. "
            "It opened to the public in a ceremony. The tower has a roof height of 1,250 feet (380 m) and "
            "stands 1,454 feet tall."}]
    f = select_facts("how tall is the tower?", res, budget=160)
    assert "1,454 feet" in f and "skyscraper" in f and "designed" not in f


def test_guard_drops_only_the_invented_clause_and_falls_back_to_a_fact():
    from planner.planner import number_guard_detail
    facts = "Wikipedia 'Tower': The Tower is a 102-story skyscraper. It has a roof height of 1,250 feet (380 m)."
    q = "how tall is the tower?"
    kept, removed = number_guard_detail("The Tower is 102 stories high, 334 feet tall.", facts, q)
    assert kept == "The Tower is 102 stories high." and removed[0][1] == ["334"]
    kept, _ = number_guard_detail("It is 334 feet tall.", facts, q)
    assert "1,250 feet" in kept


def test_new_icons_exist_and_fit_the_matrix():
    from skills.font import ICON_NAMES, ICONS, icon_frame
    assert "building" in ICON_NAMES and all(len(icon_frame(n)) == 104 for n in ICON_NAMES)
    assert set(ICON_NAMES) <= set(ICONS)


def test_do_it_again_replays_the_last_plan(gc):
    gc.submit("again")                                              # nothing yet
    assert wait_for(lambda: any("nothing to repeat" in p for _, p in gc.sent))
    gc.submit("Type Hi on the LED matrix and then send back on MQTT on topic luca message 'I am done'")
    assert wait_for(lambda: sum(p == "I am done" for t, p in gc.sent if t == "luca") == 1)
    calls = gc.llm.calls if hasattr(gc.llm, "calls") else None
    gc.submit("do it again")
    assert wait_for(lambda: sum(p == "I am done" for t, p in gc.sent if t == "luca") == 2)
    if calls is not None:
        assert gc.llm.calls == calls                                # replayed, no LLM call
    assert gc.AGAIN.match("one more time please") and not gc.AGAIN.match("spin again and again")



def test_truncated_think_is_a_prefix_not_a_mismatch():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tools" / "finetune"))
    from gen_data import think_for
    steps = [{"skill": s} for s in ["turn_by", "dance", "stop", "turn_by", "dance_style", "matrix_icon"]]
    t = think_for(steps)
    assert t.endswith("...") and len(t) <= 60 and "dance_sty" not in t.replace("dance_style", "")
