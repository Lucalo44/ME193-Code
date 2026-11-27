"""All knobs in one place. Environment variables override for laptop/sim runs."""
import os

NAME = "luca"

# --- MQTT (assignment: MQTT over WebSockets) ---------------------------------
BROKER_HOST = os.environ.get("GC_BROKER", "test.mosquitto.org")
BROKER_PORT = int(os.environ.get("GC_BROKER_PORT", "8080"))
BROKER_WS_PATH = "/mqtt"                       # spike: "/" and "/mqtt" both work
TOPIC_BASE = f"ME193/q/{NAME}"
TOPIC_CMD = f"{TOPIC_BASE}/cmd"
TOPIC_STATUS = f"{TOPIC_BASE}/status"          # retained
TOPIC_REPLY = f"{TOPIC_BASE}/reply"            # default reply topic
TOPIC_EVENTS = f"{TOPIC_BASE}/events"
STATUS_PERIOD_S = 5.0

# Topics a plan may publish to. "reply" in a plan is an alias for the job's
# reply topic. Anything ending in /cmd is always refused (no command loops,
# no injecting commands into classmates' robots).
PUBLISH_ALLOW_EXACT = {NAME, TOPIC_REPLY, TOPIC_EVENTS}
PUBLISH_ALLOW_PREFIX = (f"{TOPIC_BASE}/",)
PUBLISH_DENY_SUFFIX = ("/cmd",)

# --- untrusted-input limits ---------------------------------------------------
MAX_CMD_CHARS = 300
MIN_JOB_INTERVAL_S = 2.0
QUEUE_CAP = 5

# --- LLM ------------------------------------------------------------------------
LLM_URL = os.environ.get("GC_LLM_URL", "http://llamacpp-models-runner:9999")
LLM_MODEL = os.environ.get("GC_LLM_MODEL", "custom/gc-planner-v2-Q4_0")   # fine-tuned v2: 98% held-out (docs/EVAL.md)
LLM_MODELS = ["custom/gc-planner-v2-Q4_0", "custom/gc-planner-Q4_0"]   # dashboard picker: v2, v1 (base Qwen3-1.7B removed)
THINK_MAX_CHARS = int(os.environ.get("GC_THINK_CHARS", "60"))   # 0 = no think field
MAX_STEPS = 8
# "think": the model names the skills it will use in a short think, then writes steps
#          (repair if the steps don't match the names). Best held-out score.
# "parts": the model copies the command's clauses into parts, one step per part.
PLAN_STYLE = os.environ.get("GC_PLAN_STYLE", "think")
# The fine-tuned planner (tools/finetune/). Selecting it switches the planner to the "ft"
# prompt style (no few-shot examples); install with `tools/deploy.py --model <file.gguf>`.
FT_MODEL = "custom/gc-planner-Q4_0"     # App Lab names it after its folder: models/llamacpp/custom/


def style_for(model):
    # Every fine-tuned planner (v1, v2, ...) was trained on the "ft" prompt; matching v1's
    # exact name alone sent v2 the few-shot prompt in its first eval.
    if model == FT_MODEL or str(model).split("/")[-1].startswith("gc-planner"):
        return "ft"
    return "think" if PLAN_STYLE == "ft" else PLAN_STYLE
LLM_MAX_TOKENS = 220
LLM_TIMEOUT_S = 180
LLM_TOP_K = 5

# --- hardware -------------------------------------------------------------------
SIM = os.environ.get("GC_SIM", "0") == "1"
HEARTBEAT_S = 0.2
WHEEL_MAX = 255                 # sketch's MAX_SPEED
CAR_DEADMAN_S = 0.6             # drive() refresh needed or the wheels stop

# LEGO Double Motor: reached through lego_service/lego_server.py on the board HOST
# (the app container has no D-Bus/BlueZ access). In the container the host is the
# Docker bridge gateway; on the laptop it's localhost (relay mode).
LEGO_ENABLED = os.environ.get("GC_LEGO", "1") == "1"
LEGO_HOST = os.environ.get("GC_LEGO_HOST", "")      # "" = auto (bridge gateway)
LEGO_PORT = int(os.environ.get("GC_LEGO_PORT", "8770"))
LEGO_CARD = os.environ.get("GC_LEGO_CARD", "0998")
LEGO_CARD_COLOR = os.environ.get("GC_LEGO_CARD_COLOR", "purple")
LEGO_BUILD = os.environ.get("GC_LEGO_BUILD", "arms")   # "arms" (A/B = two arms) or "wheels"

# --- dance ----------------------------------------------------------------------
BPM_RANGE = (60, 160)
MAX_SECTIONS = 4
MAX_SECTION_BARS = 4
MAX_DANCE_S = 60
MAX_LOOP_S = 300               # a looped dance ("loop it", "until I say stop") ends after 5 min or on STOP
MAX_LOOPS = 20
COUNT_IN_BEATS = 4              # matrix counts 4-3-2-1 before the first beat (one bar, like a band)
LOOKAHEAD_S = 0.15

# --- dashboard ---
WEB_TLS = os.environ.get("GC_WEB_TLS", "1") == "1"   # https on :7000 (needed for the mic)

# --- viz ------------------------------------------------------------------------
VIZ_QUEUE = 500
