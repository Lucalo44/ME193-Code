"""Cache-friendly prompt.

FIXED PREFIX (system message): byte-identical on every call, warmed at boot,
so llama.cpp reuses its KV cache and only the user's command is prefilled.
DYNAMIC SUFFIX (user message): just the command. No live sensor state goes in
the prompt; plans read sensors through skills.
"""
import json

import config
import skills
from skills.dance_skill import DANCE_CATALOG_EXTRA

HEAD = ("You plan actions for a dancing robot car and its LEGO partner robot.\n"
        "Reply with ONE JSON plan. Skills:")

RULES_PARTS = [
    "First copy the command's actions into parts, in the user's words, one action per part. Skip actions the user says not to do.",
    "Then write exactly one step per part, in order. Nothing extra.",
    "mqtt_publish only when the user asks to tell, send, reply or report. A \"done\" reply is sent automatically.",
    "Publish to the topic the user names, else topic \"reply\". {imu.yaw} {imu.pitch} {imu.roll} in a message are filled in.",
    "Questions or chat: answer in words. Impossible or unsafe requests: no parts, no steps. Never publish to a topic ending in /cmd.",
]

EXAMPLES_PARTS = [
    ("show a heart",
     {"parts": ["show a heart"],
      "steps": [{"skill": "matrix_icon", "args": {"name": "heart"}}]}),
    ("write GO, then drive forward for 2 seconds",
     {"parts": ["write GO", "drive forward for 2 seconds"],
      "steps": [{"skill": "matrix_text", "args": {"text": "GO"}},
                {"skill": "drive", "args": {"speed": 60, "seconds": 2}}]}),
    ("what are your yaw and roll? answer on ME193/q/luca/info",
     {"parts": ["what are your yaw and roll? answer on ME193/q/luca/info"],
      "steps": [{"skill": "mqtt_publish", "args": {"topic": "ME193/q/luca/info",
                                                   "message": "yaw {imu.yaw} roll {imu.roll}"}}]}),
    ("dance 2 bars at 120 bpm with the lego bot mirroring you, then tell luca finished",
     {"parts": ["dance 2 bars at 120 bpm with the lego bot mirroring you", "tell luca finished"],
      "steps": [{"skill": "dance", "args": {"bpm": 120, "sections": [
          {"bars": 2, "mode": "mirror", "car": "wiggle", "lego": "pump"}]}},
                {"skill": "mqtt_publish", "args": {"topic": "luca", "message": "finished"}}]}),
    ("what is 7 times 6?",
     {"parts": ["what is 7 times 6?"],
      "steps": [{"skill": "answer", "args": {"text": "7 times 6 is 42."}}]}),
    ("make me a sandwich",
     {"parts": [], "steps": []}),
]


# ---- "think" style (the first version, plus the verb catalog) ----
HEAD_THINK = ("You plan actions for a dancing robot car and its LEGO partner robot.\n"
              "Reply with ONE JSON plan. In think, name the skills you will use, in order. Skills:")

RULES_THINK = [
    "Do only what is asked, in order.",
    "Reply with mqtt_publish to the topic the user names, else topic \"reply\".",
    "To report a value, read it first, then use its {placeholder} in the message.",
    "Questions or chat: use answer. Impossible or unsafe requests: no steps. Never publish to a topic ending in /cmd.",
]

EXAMPLES_THINK = [
    ("show a heart and then tell me done on topic luca",
     {"think": "matrix_icon, then mqtt_publish to luca",
      "steps": [{"skill": "matrix_icon", "args": {"name": "heart"}},
                {"skill": "mqtt_publish", "args": {"topic": "luca", "message": "done"}}]}),
    ("what are your yaw and roll? answer on ME193/q/luca/info",
     {"think": "read_imu, then mqtt_publish the values",
      "steps": [{"skill": "read_imu", "args": {}},
                {"skill": "mqtt_publish", "args": {"topic": "ME193/q/luca/info",
                                                   "message": "yaw {imu.yaw} roll {imu.roll}"}}]}),
    ("write GO, drive forward for 2 seconds, then say arrived",
     {"think": "matrix_text, drive, then mqtt_publish to reply",
      "steps": [{"skill": "matrix_text", "args": {"text": "GO"}},
                {"skill": "drive", "args": {"speed": 60, "seconds": 2}},
                {"skill": "mqtt_publish", "args": {"topic": "reply", "message": "arrived"}}]}),
    ("dance 2 bars at 120 bpm with the lego bot mirroring you",
     {"think": "dance, one mirror section",
      "steps": [{"skill": "dance", "args": {"bpm": 120, "sections": [
          {"bars": 2, "mode": "mirror", "car": "shimmy", "lego": "wave"}]}}]}),
    ("what is 7 times 6?",
     {"think": "answer", "steps": [{"skill": "answer", "args": {"text": "7 times 6 is 42."}}]}),
    ("make me a sandwich",
     {"think": "no skill for that", "steps": []}),
]

STYLES = {"parts": (HEAD, RULES_PARTS, EXAMPLES_PARTS, "parts"),
          "think": (HEAD_THINK, RULES_THINK, EXAMPLES_THINK, "think")}


def _example_json(plan, think_chars, key="parts"):
    p = dict(plan)
    if think_chars <= 0:
        p.pop(key, None)
    return json.dumps(p, separators=(",", ":"))


def segments(think_chars=60, style=None):
    """Ordered (kind, text) pieces of the fixed prefix, for the token-chip view."""
    head, rules_l, examples, key = STYLES[style or config.PLAN_STYLE]
    catalog = skills.catalog_text(style or config.PLAN_STYLE) + "\n" + DANCE_CATALOG_EXTRA
    rules = "Rules:\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules_l))
    segs = [("system", head + "\n"), ("catalog", catalog + "\n"), ("rules", rules)]
    for c, p in examples:
        segs.append(("examples", c))
        segs.append(("examples", _example_json(p, think_chars, key)))
    return segs


def system_prompt(think_chars=60, style=None):
    return "".join(t for k, t in segments(think_chars, style) if k != "examples")


def messages(command, think_chars=60, style=None):
    """System prompt, then the few-shot examples as real chat turns (small models
    copy demonstrated turns far better than examples pasted into the system
    text), then the command. Everything before the last message is fixed."""
    style = style or config.PLAN_STYLE
    _, _, examples, key = STYLES[style]
    msgs = [{"role": "system", "content": system_prompt(think_chars, style)}]
    for c, p in examples:
        msgs.append({"role": "user", "content": c})
        msgs.append({"role": "assistant", "content": _example_json(p, think_chars, key)})
    msgs.append({"role": "user", "content": command})
    return msgs
