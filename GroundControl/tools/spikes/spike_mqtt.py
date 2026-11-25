#!/usr/bin/env python3
"""Phase 0 spike: MQTT over WebSockets to ws://test.mosquitto.org:8080.

Subscribes to a random test topic, publishes to it and times the round trip,
for each candidate WebSocket path. Needs paho-mqtt (present in the app image).

    python3 spike_mqtt.py
"""
import threading
import time
import uuid

import paho.mqtt.client as mqtt

HOST, PORT = "test.mosquitto.org", 8080
TOPIC = f"ME193/q/luca/spike/{uuid.uuid4().hex[:6]}"


def trial(path):
    got = threading.Event()
    t_sent = {}

    def on_connect(c, u, flags, rc, props=None):
        c.subscribe(TOPIC, qos=1)

    def on_subscribe(c, u, mid, rc, props=None):
        t_sent["t"] = time.time()
        c.publish(TOPIC, "ping", qos=1)

    def on_message(c, u, msg):
        t_sent["rtt"] = (time.time() - t_sent["t"]) * 1000
        got.set()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"gc-spike-{uuid.uuid4().hex[:8]}",
                    transport="websockets")
    c.ws_set_options(path=path)
    c.on_connect, c.on_subscribe, c.on_message = on_connect, on_subscribe, on_message
    t0 = time.time()
    try:
        c.connect(HOST, PORT, keepalive=20)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  path={path!r:8} connect error {e!r}")
        return False
    c.loop_start()
    ok = got.wait(15)
    c.loop_stop()
    c.disconnect()
    if ok:
        print(f"PASS  path={path!r:8} connect+sub+pub round trip, total {(time.time() - t0) * 1000:.0f} ms, "
              f"pub->msg {t_sent['rtt']:.0f} ms")
    else:
        print(f"FAIL  path={path!r:8} no message within 15 s")
    return ok


print(f"MQTT over WebSockets ws://{HOST}:{PORT}  topic {TOPIC}")
for p in ["/mqtt", "/"]:
    trial(p)
