"""Runs on the UNO Q's Linux side as the Python half of the sumo robot app.
It connects the sketch to MQTT in both directions:

  paddle limit switches -> <base>/switch   {"pressed", "left", "right"}
  <base>/drive {"left", "right"}  -> wheel motors (-255..255)

The decision tree runs on the laptop (sumo wrestling/sumo.py), which listens
on <base>/switch and publishes <base>/drive. Broker and TOPIC_BASE must match
that script.

The switch state is published as soon as it changes and as a heartbeat every
SWITCH_HEARTBEAT seconds. The laptop treats a switch it hasn't heard from in
1 s as released, so a dead UnoQ can't leave it stuck "pressed".

Safety: if no drive message arrives for CONTROL_TIMEOUT seconds (laptop
crashed, WiFi dropped), the wheels are stopped here regardless of the last
command.
"""

import json
import threading
import time

import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge

MQTT_BROKER = "test.mosquitto.org"  # must match BROKER in sumo.py
MQTT_PORT = 1883
TOPIC_BASE = "ME193/Luca/sumo"      # must match TOPIC_BASE in sumo.py
TOPIC_SWITCH = f"{TOPIC_BASE}/switch"
TOPIC_DRIVE = f"{TOPIC_BASE}/drive"

MAX_SPEED = 255          # must match MAX_SPEED in the sketch
LOOP_INTERVAL = 0.01     # seconds -- switch polling rate
SWITCH_HEARTBEAT = 0.2   # seconds -- resend switch state at least this often
CONTROL_TIMEOUT = 0.75   # seconds -- stop the wheels if drive goes quiet

_state_lock = threading.Lock()
_drive = (0, 0)
_drive_seen = 0.0


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected (rc={rc}), subscribing to {TOPIC_DRIVE!r}")
    client.subscribe(TOPIC_DRIVE)


def on_disconnect(client, userdata, rc):
    print(f"[mqtt] disconnected (rc={rc})")


def on_message(client, userdata, msg):
    global _drive, _drive_seen
    try:
        data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        left = max(-MAX_SPEED, min(MAX_SPEED, int(data["left"])))
        right = max(-MAX_SPEED, min(MAX_SPEED, int(data["right"])))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return
    with _state_lock:
        _drive = (left, right)
        _drive_seen = time.monotonic()


client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.reconnect_delay_set(min_delay=1, max_delay=10)
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
client.loop_start()

_sent_drive = None
_last_switches = None
_last_switch_publish = 0.0


def loop():
    global _sent_drive, _last_switches, _last_switch_publish
    now = time.monotonic()

    # --- wheels -------------------------------------------------------------
    with _state_lock:
        drive, drive_seen = _drive, _drive_seen
    if now - drive_seen > CONTROL_TIMEOUT:
        drive = (0, 0)
    if drive != _sent_drive:
        Bridge.call("drive", drive[0], drive[1])
        _sent_drive = drive

    # --- limit switches -----------------------------------------------------
    bits = int(Bridge.call("read_switches") or 0)
    switches = (bool(bits & 1), bool(bits & 2))
    if switches != _last_switches or now - _last_switch_publish >= SWITCH_HEARTBEAT:
        left, right = switches
        client.publish(TOPIC_SWITCH, json.dumps(
            {"pressed": left or right, "left": left, "right": right}))
        if switches != _last_switches:
            print(f"[switch] left={'ON ' if left else 'off'} right={'ON ' if right else 'off'}")
        _last_switches = switches
        _last_switch_publish = now

    time.sleep(LOOP_INTERVAL)


App.run(user_loop=loop)
