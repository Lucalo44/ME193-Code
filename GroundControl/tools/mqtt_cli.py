#!/usr/bin/env python3
"""Talk to Ground Control over the public broker (MQTT over WebSockets).

    venv/bin/python tools/mqtt_cli.py send "Type Hi on the LED matrix and then send back on topic luca 'I am done'"
    venv/bin/python tools/mqtt_cli.py send "/status"
    venv/bin/python tools/mqtt_cli.py run-plan tools/plans/hi_then_done.json
    venv/bin/python tools/mqtt_cli.py watch               # replies, status, events, and topic 'luca'
    venv/bin/python tools/mqtt_cli.py status-latency      # how fast does /status answer?

`send` waits for the reply (on the reply topic or on 'luca') and prints the
round-trip time.
"""
import json
import sys
import threading
import time
import uuid
from pathlib import Path

import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import config  # noqa: E402

WATCH = [config.NAME, f"{config.TOPIC_BASE}/#"]   # the wildcard covers reply/status/events


def client(on_message=None):
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"gc-cli-{uuid.uuid4().hex[:6]}",
                    transport="websockets")
    c.ws_set_options(path=config.BROKER_WS_PATH)
    ready = threading.Event()

    def on_connect(cl, u, f, rc, p=None):
        for t in WATCH:
            cl.subscribe(t, qos=1)
        ready.set()
    c.on_connect = on_connect
    if on_message:
        c.on_message = on_message
    c.connect(config.BROKER_HOST, config.BROKER_PORT, keepalive=30)
    c.loop_start()
    ready.wait(10)
    time.sleep(0.3)
    return c


def send(payload, timeout=180):
    got = threading.Event()
    t0 = [0.0]
    want_status = payload.strip().lower() in ("/status", "status")

    def on_message(c, u, msg):
        if msg.topic not in (config.TOPIC_REPLY, config.NAME):
            return
        text = msg.payload.decode(errors="replace")
        if text.startswith("{") and not want_status:
            return                      # someone else's /status answer
        print(f"[{time.time() - t0[0]:6.2f}s] {msg.topic}: {text}")
        low = text.lower()
        if want_status or msg.topic == config.NAME or msg.topic == config.TOPIC_REPLY \
                or "done" in low or "sorry" in low:
            got.set()
    c = client(on_message)
    t0[0] = time.time()
    c.publish(config.TOPIC_CMD, payload, qos=1)
    print(f"sent to {config.TOPIC_CMD}: {payload[:200]}")
    if not got.wait(timeout):
        print(f"no reply within {timeout}s")
    time.sleep(1.0)        # catch a trailing 'luca' message
    c.loop_stop()


def watch():
    def on_message(c, u, msg):
        p = msg.payload.decode(errors="replace")
        if msg.topic == config.TOPIC_STATUS:
            try:
                s = json.loads(p)
                p = f"{s.get('state')}: {s.get('doing')} (queue {s.get('queue')}, uptime {s.get('uptime_s')} s)"
            except json.JSONDecodeError:
                pass
        print(f"{time.strftime('%H:%M:%S')} {msg.topic}: {p[:300]}")
    client(on_message)
    print("watching (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def status_latency(n=5):
    times = []
    ev = threading.Event()

    def on_message(c, u, msg):
        if msg.topic == config.TOPIC_REPLY and msg.payload.startswith(b"{"):
            ev.set()
    c = client(on_message)
    for _ in range(n):
        ev.clear()
        t = time.time()
        c.publish(config.TOPIC_CMD, "/status", qos=1)
        if ev.wait(10):
            times.append((time.time() - t) * 1000)
        time.sleep(0.5)
    print(f"/status round trip over the public broker: {[round(x) for x in times]} ms")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "send":
        send(" ".join(sys.argv[2:]))
    elif cmd == "run-plan":
        plan = json.loads(Path(sys.argv[2]).read_text())
        send(json.dumps({"plan": plan, "text": f"run-plan {Path(sys.argv[2]).name}"}))
    elif cmd == "watch":
        watch()
    elif cmd == "status-latency":
        status_latency()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
