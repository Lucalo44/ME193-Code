"""MQTT over WebSockets: commands in, replies/status/events out.

Fast paths handled right here on the paho thread, never queued, never LLM:
  "/status" or "status" -> immediate status reply + retained status refresh
  "stop", "/stop", "estop", "freeze" -> e-stop callback (cancels jobs, halts all robots)
Everything else is rate-limited, length-capped and handed to on_command.

Payload: plain text, or JSON {"text": "...", "id": "...", "reply_to": "...", "source": "voice"}.
"""
import json
import threading
import time
import uuid

import config

try:
    import paho.mqtt.client as mqtt
except ImportError:  # laptop tests without paho
    mqtt = None

STATUS_WORDS = {"/status", "status", "status?"}
STOP_WORDS = {"stop", "/stop", "estop", "e-stop", "freeze", "stop!", "halt"}


def topic_allowed(topic: str) -> bool:
    if not topic or len(topic) > 100 or any(c in topic for c in "#+\0"):
        return False
    if topic.endswith(config.PUBLISH_DENY_SUFFIX):
        return False
    return topic in config.PUBLISH_ALLOW_EXACT or topic.startswith(config.PUBLISH_ALLOW_PREFIX)


def parse_payload(raw: bytes):
    """-> dict(text, reply_to, id, source) or None."""
    s = raw.decode("utf-8", errors="replace").strip()
    if not s:
        return None
    d = {"text": s, "reply_to": None, "id": None, "source": "mqtt", "plan": None}
    if s.startswith("{"):
        try:
            j = json.loads(s)
            if isinstance(j, dict) and isinstance(j.get("plan"), dict):
                # Hand-written plan (tools/mqtt_cli.py run-plan): skips the LLM, but is
                # validated, clamped and topic-whitelisted exactly like LLM output.
                d["plan"] = j["plan"]
                d["text"] = str(j.get("text") or "hand-written plan")
                d["reply_to"] = j.get("reply_to") if isinstance(j.get("reply_to"), str) else None
            elif isinstance(j, dict) and isinstance(j.get("text"), str):
                d["text"] = j["text"].strip()
                d["reply_to"] = j.get("reply_to") if isinstance(j.get("reply_to"), str) else None
                d["id"] = str(j.get("id"))[:32] if j.get("id") is not None else None
                if j.get("source") in ("voice", "mqtt", "ui"):
                    d["source"] = j["source"]
        except json.JSONDecodeError:
            pass
    d["text"] = d["text"][: config.MAX_CMD_CHARS]
    if d["reply_to"] and not topic_allowed(d["reply_to"]):
        d["reply_to"] = None
    return d


class MqttIO:
    def __init__(self, status, on_command, on_estop):
        self.status = status
        self.on_command = on_command
        self.on_estop = on_estop
        self._last_job_t = 0.0
        self._lock = threading.Lock()
        self.connected = False
        self.client = None
        self.on_publish_hook = None      # viz: called with (topic, payload)

    # ---- lifecycle ----
    def start(self):
        if mqtt is None:
            print("[mqtt] paho not installed; MQTT disabled", flush=True)
            return
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                        client_id=f"gc-{config.NAME}-{uuid.uuid4().hex[:8]}",
                        transport="websockets")
        c.ws_set_options(path=config.BROKER_WS_PATH)
        c.reconnect_delay_set(min_delay=1, max_delay=30)
        c.will_set(config.TOPIC_STATUS, json.dumps({"alive": False, "state": "OFFLINE"}),
                   qos=1, retain=True)
        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_message = self._on_message
        self.client = c
        c.connect_async(config.BROKER_HOST, config.BROKER_PORT, keepalive=30)
        c.loop_start()

    def stop(self):
        if self.client:
            self.publish_status(self.status.snapshot() | {"alive": False, "state": "OFFLINE"})
            self.client.loop_stop()

    # ---- paho callbacks ----
    def _on_connect(self, c, u, flags, rc, props=None):
        print(f"[mqtt] connected rc={rc}; subscribing {config.TOPIC_CMD}", flush=True)
        c.subscribe(config.TOPIC_CMD, qos=1)
        self.connected = True
        self.status.update(mqtt=True)

    def _on_disconnect(self, c, u, flags, rc, props=None):
        print(f"[mqtt] disconnected rc={rc}", flush=True)
        self.connected = False
        self.status.update(mqtt=False)

    def _on_message(self, c, u, msg):
        d = parse_payload(msg.payload)
        if d:
            self.handle(d)

    # ---- routing (also used by the dashboard's command box) ----
    def handle(self, d):
        """Returns a short string describing what happened (for the UI)."""
        text = d["text"]
        low = text.lower().strip().rstrip(".!")
        reply_to = d.get("reply_to") or config.TOPIC_REPLY
        if d.get("plan") is None and low in STATUS_WORDS:
            snap = self.status.snapshot()
            self.publish(reply_to, json.dumps(snap))
            self.publish_status(snap)
            return "status"
        if low in STOP_WORDS:
            self.on_estop(source=d.get("source", "mqtt"))
            self.publish(reply_to, "stopped")
            return "estop"
        with self._lock:
            now = time.time()
            if d.get("source") != "ui" and now - self._last_job_t < config.MIN_JOB_INTERVAL_S:
                self.publish(reply_to, "Sorry, too many commands. Try again in a moment.")
                return "rate_limited"
            self._last_job_t = now
        ok, why = self.on_command(text, d.get("source", "mqtt"), reply_to, plan=d.get("plan"))
        if not ok:
            self.publish(reply_to, f"Sorry, I can't take that right now: {why}")
            return why
        return "queued"

    # ---- publishing ----
    def publish(self, topic, payload, qos=1, retain=False):
        if self.on_publish_hook:
            self.on_publish_hook(topic, payload)
        if self.client is None:
            print(f"[mqtt:sim] {topic} <- {payload[:200]}", flush=True)
            return
        self.client.publish(topic, payload, qos=qos, retain=retain)

    def publish_status(self, snap):
        if self.client is not None:
            self.client.publish(config.TOPIC_STATUS, json.dumps(snap), qos=1, retain=True)

    def publish_event(self, kind, payload):
        if self.client is not None:
            self.client.publish(config.TOPIC_EVENTS, json.dumps({"kind": kind, **payload}), qos=0)
