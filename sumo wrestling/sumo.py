#!/usr/bin/env python3
"""
sumo.py -- laptop brain for the sumo robot.

Hardware
--------
    LEGO Controller      teleop: left/right levers = tank drive,
                         controller button = start match / manual flip
    LEGO Color Sensor    autonomous: mounted at the REAR of the robot,
                         looking down, watching for the blue ring tape
    LEGO Double Motor    the flipper paddle
    UnoQ                 reads the paddle limit switches and drives the
                         car's wheel motors, both over MQTT

Everything talks to this script: the three LEGO devices over Bluetooth,
the UnoQ over MQTT. Every loop the sensors are read into an Inputs
snapshot, decide() (the decision tree) picks an action, and the action is
carried out.

Decision tree (decide() below, checked top to bottom)
-----------------------------------------------------
    match not started?               -> WAITING      wheels stopped
    pushed out of the ring?          -> LOST         stop, play sad song
    rear sensor on the blue tape?    -> ESCAPE       drive forward, ignore levers
    limit switch hit, flipper ready? -> AUTO_FLIP    fire flipper, keep driving
    button pressed, flipper ready?   -> MANUAL_FLIP  fire flipper, keep driving
    otherwise                        -> TELEOP       wheels follow the levers

"Out of the ring" is detected as: the rear sensor stays on the tape for
OUT_TAPE_S even though ESCAPE is driving forward at full power, i.e. we
are being pushed over the line. One color sensor can't tell the floor
inside the ring from the floor outside it, so if the robot is shoved
across the tape faster than OUT_TAPE_S it won't notice -- tune OUT_TAPE_S
in the real ring.

MQTT messages (TOPIC_BASE defaults to ME193/Luca/sumo)
------------------------------------------------------
    <base>/switch   UnoQ -> laptop   {"pressed": bool, "left": bool,
                                     "right": bool} (plain "1"/"0" also
                                     works), on change + every 0.2 s.
                                     Sent by ArduinoUNOq/sumo-unoq.
    <base>/drive    laptop -> UnoQ   {"left": int, "right": int}, -255..255,
                                     same format as mqtt-minifig-drive. The
                                     UnoQ should stop its motors if this goes
                                     quiet for ~0.75 s.
    <base>/state    laptop -> anyone {"state": "TELEOP", "color": 3, ...}

Usage
-----
    source venv/bin/activate
    python sumo.py                            # connects to devices with our card
    python sumo.py --card 1234 --card-color red   # a different Connection Card
    python sumo.py --broker 192.168.1.20      # local mosquitto, lower latency

Press the controller button to start a match (and again to re-arm after a
loss). Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass

import legoeducation as le
import paho.mqtt.client as mqtt


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC_BASE = "ME193/Luca/sumo"

# Our Connection Card. Tap it onto all three LEGO devices; each one is then
# found by card + device type, so other teams' hubs are ignored.
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "0998"      # a string, to keep the leading zero

LOOP_PERIOD = 0.02          # s -- decision tree runs at 50 Hz
NOTIFY_MS = 20              # ms -- how often LEGO sensors report (15..1000)

# Drive (UnoQ wheel motors)
MAX_DRIVE = 255             # must match MAX_SPEED in the UnoQ sketch
LEVER_DEADBAND = 8          # % -- ignore tiny lever wobble around center
INVERT_LEFT = False         # flip if a wheel drives backwards
INVERT_RIGHT = False
DRIVE_HEARTBEAT = 0.1       # s -- resend drive at least this often

# Ring edge (Color Sensor)
TAPE_COLORS = {le.LEGO_COLOR_BLUE, le.LEGO_COLOR_AZURE}  # blue tape can read as either
TAPE_DEBOUNCE_S = 0.04      # must see tape this long before it counts
ESCAPE_DRIVE = MAX_DRIVE    # forward speed while backing off the tape
OUT_TAPE_S = 0.6            # on tape this long despite ESCAPE -> we're out

# Flipper (Double Motor)
FLIP_DEGREES = 120          # how far the paddle swings up
FLIP_SPEED = 100            # % -- swing up as hard as possible
RETURN_SPEED = 50           # % -- lower the paddle gently
FLIP_COOLDOWN_S = 0.8       # s -- after returning, before it can fire again
FLIP_MIRRORED = True        # the two sides drive the paddle from opposite
                            # sides (mirror image). Set False if both outputs
                            # face the same way and the sides fight each other.

# MQTT limit switch
SWITCH_STALE_S = 1.0        # no UnoQ message this long -> treat as released

# Sad song: (frequency Hz, seconds) -- "wah wah wah waaah"
SAD_SONG = [(392, 0.45), (370, 0.45), (349, 0.45), (330, 1.4)]


# --------------------------------------------------------------------------
# Decision tree
# --------------------------------------------------------------------------

@dataclass
class Inputs:
    match_on: bool        # button pressed to start the match
    out: bool             # RingMonitor says we've been pushed out
    on_tape: bool         # rear color sensor is on the blue tape
    switch: bool          # paddle limit switch pressed (UnoQ via MQTT)
    button_tap: bool      # controller button was just pressed
    flipper_ready: bool   # flipper idle and cooled down
    lever_left: int       # controller levers, -100..100
    lever_right: int


@dataclass
class Decision:
    state: str
    left: int = 0         # wheel speeds for the UnoQ, -255..255
    right: int = 0
    fire: bool = False    # swing the flipper this tick


def levers_to_drive(lever_left: int, lever_right: int) -> tuple[int, int]:
    """Tank drive: each lever (-100..100 %) sets one wheel (-255..255)."""
    def scale(percent, invert):
        if abs(percent) < LEVER_DEADBAND:
            return 0
        speed = int(round(percent * MAX_DRIVE / 100))
        return -speed if invert else speed
    return scale(lever_left, INVERT_LEFT), scale(lever_right, INVERT_RIGHT)


def decide(i: Inputs) -> Decision:
    if not i.match_on:
        return Decision("WAITING")
    if i.out:
        return Decision("LOST")
    if i.on_tape:
        return Decision("ESCAPE", ESCAPE_DRIVE, ESCAPE_DRIVE)

    left, right = levers_to_drive(i.lever_left, i.lever_right)
    if i.flipper_ready and i.switch:
        return Decision("AUTO_FLIP", left, right, fire=True)
    if i.flipper_ready and i.button_tap:
        return Decision("MANUAL_FLIP", left, right, fire=True)
    return Decision("TELEOP", left, right)


# --------------------------------------------------------------------------
# Sensing
# --------------------------------------------------------------------------

class RingMonitor:
    """Turns raw color readings into on_tape / out."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._tape_since = None
        self.on_tape = False
        self.out = False

    def update(self, color, now: float):
        if color in TAPE_COLORS:
            if self._tape_since is None:
                self._tape_since = now
            on_for = now - self._tape_since
            self.on_tape = on_for >= TAPE_DEBOUNCE_S
            if on_for >= OUT_TAPE_S:
                self.out = True  # latched until the next match
        else:
            self._tape_since = None
            self.on_tape = False


class SwitchLink:
    """MQTT: receives the paddle limit switch, sends wheel speeds and state."""

    def __init__(self, broker: str, port: int, topic_base: str):
        self.topic_switch = f"{topic_base}/switch"
        self.topic_drive = f"{topic_base}/drive"
        self.topic_state = f"{topic_base}/state"
        self._pressed = False
        self._hit = False         # pressed since the main loop last asked
        self._last_seen = 0.0
        self._lock = threading.Lock()

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=10)
        self._client.connect(broker, port, keepalive=30)
        self._client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"[mqtt] connected ({reason_code}), listening on {self.topic_switch}")
        client.subscribe(self.topic_switch)

    def _on_message(self, client, userdata, msg):
        text = msg.payload.decode("utf-8", errors="replace").strip().lower()
        try:
            data = json.loads(text)
            pressed = bool(data["pressed"]) if isinstance(data, dict) else bool(data)
        except (ValueError, KeyError, TypeError):
            pressed = text in ("1", "true", "pressed", "on")
        with self._lock:
            self._pressed = pressed
            self._hit = self._hit or pressed
            self._last_seen = time.monotonic()

    def switch(self) -> tuple[bool, bool]:
        """(pressed, fresh). Pressed if held now OR hit since the last call,
        so a tap that is released before the next loop still counts. A stale
        switch reads as not pressed."""
        with self._lock:
            fresh = time.monotonic() - self._last_seen <= SWITCH_STALE_S
            pressed = (self._pressed or self._hit) and fresh
            self._hit = False
            return pressed, fresh

    def drive(self, left: int, right: int):
        self._client.publish(self.topic_drive, json.dumps({"left": left, "right": right}))

    def state(self, payload: dict):
        self._client.publish(self.topic_state, json.dumps(payload))

    def close(self):
        self.drive(0, 0)
        time.sleep(0.1)  # let the stop message go out
        self._client.loop_stop()
        self._client.disconnect()


# --------------------------------------------------------------------------
# Actuators
# --------------------------------------------------------------------------

class Flipper:
    """Swings the paddle up and back down on a background thread, so the
    main loop keeps driving while the flip happens."""

    def __init__(self, motor: le.DoubleMotor):
        self._motor = motor
        self._busy = False
        self._done_at = 0.0
        self._lock = threading.Lock()

    def ready(self) -> bool:
        with self._lock:
            return not self._busy and time.monotonic() - self._done_at >= FLIP_COOLDOWN_S

    def fire(self):
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._swing, daemon=True).start()

    def _move(self, direction_up: bool, speed: int):
        if FLIP_MIRRORED:
            direction = (le.MOVEMENT_MOVE_DIRECTION_FORWARD if direction_up
                         else le.MOVEMENT_MOVE_DIRECTION_BACKWARD)
            self._motor.movement_move_for_degrees(FLIP_DEGREES, direction=direction, speed=speed)
        else:
            direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if direction_up
                         else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
            self._motor.motor_run_for_degrees(FLIP_DEGREES, direction=direction,
                                              motor=le.MOTOR_BOTH, speed=speed)

    def _swing(self):
        try:
            self._move(True, FLIP_SPEED)
            self._move(False, RETURN_SPEED)
        except Exception as e:
            print(f"[flip] failed: {e}")
        finally:
            with self._lock:
                self._busy = False
                self._done_at = time.monotonic()

    def stop(self):
        try:
            self._motor.movement_stop(blocking=False)
        except Exception as e:
            print(f"[flip] stop failed: {e}")


def play_sad_song(device):
    """Beeps the melody on a LEGO device's speaker (blocking)."""
    for freq, dur in SAD_SONG:
        device.beep(frequency=freq, count=1, blocking=False)
        time.sleep(dur)
        device.stop_beep(blocking=False)
        time.sleep(0.05)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

CARD_COLORS = {"red": le.LEGO_COLOR_RED, "yellow": le.LEGO_COLOR_YELLOW,
               "blue": le.LEGO_COLOR_BLUE, "green": le.LEGO_COLOR_GREEN,
               "purple": le.LEGO_COLOR_PURPLE, "orange": le.LEGO_COLOR_ORANGE,
               "teal": le.LEGO_COLOR_TEAL, "white": le.LEGO_COLOR_WHITE,
               "magenta": le.LEGO_COLOR_MAGENTA, "azure": le.LEGO_COLOR_AZURE}


def add_card_args(parser):
    """--card / --card-color, defaulting to our purple 0998 card."""
    default_color = next(n for n, c in CARD_COLORS.items() if c == CARD_COLOR)
    parser.add_argument("--card", default=CARD_SERIAL,
                        help=f"Connection Card serial (default {CARD_SERIAL})")
    parser.add_argument("--card-color", default=default_color, choices=CARD_COLORS,
                        help=f"Connection Card color (default {default_color})")


def connect(device, name: str, card_serial=CARD_SERIAL, card_color=CARD_COLOR):
    """Connect with retries -- the hubs sometimes report 'not ready' at first."""
    print(f"[lego] connecting to {name} (card {card_serial})...")
    for attempt in range(5):
        try:
            device.connect(card_serial=card_serial, card_color=card_color,
                           device_notification_delay=NOTIFY_MS)
            break
        except Exception as e:
            if "not ready" in str(e).lower() and attempt < 4:
                time.sleep(1)
            else:
                raise
    if not device.connected:
        raise ConnectionError(
            f"could not find {name} with card {card_serial}. Is it on, and was "
            f"the card tapped onto it? Make sure it isn't connected to another "
            f"program or the LEGO app.")
    print(f"[lego] {name} connected")
    return device


def main():
    parser = argparse.ArgumentParser(description="Sumo robot: decision tree + MQTT + LEGO")
    parser.add_argument("--broker", default=BROKER)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--topic-base", default=TOPIC_BASE)
    add_card_args(parser)
    args = parser.parse_args()

    link = SwitchLink(args.broker, args.port, args.topic_base)
    card = (args.card, CARD_COLORS[args.card_color])
    motor = connect(le.DoubleMotor(), "Double Motor (flipper)", *card)
    sensor = connect(le.ColorSensor(), "Color Sensor", *card)
    pad = connect(le.Controller(), "Controller", *card)

    ring = RingMonitor()
    flipper = Flipper(motor)

    match_on = False
    button_was_down = False
    last_state = None
    last_drive = None
    last_drive_sent = 0.0
    last_print = 0.0
    warned_stale = False

    print("\nReady. Press the controller button to start the match. Ctrl+C to quit.\n")
    try:
        while True:
            now = time.monotonic()

            # --- read sensors -------------------------------------------------
            button_down = pad.button.state == le.BUTTON_STATE_PRESSED
            button_tap = button_down and not button_was_down
            button_was_down = button_down

            color = sensor.sensor.color
            ring.update(color, now)
            switch, switch_fresh = link.switch()
            if switch_fresh:
                warned_stale = False
            elif match_on and not warned_stale:
                print("[mqtt] no limit-switch messages from the UnoQ -- auto flip disabled")
                warned_stale = True

            # Button starts a match, or re-arms after a loss.
            if button_tap and (not match_on or ring.out):
                match_on = True
                ring.reset()
                button_tap = False  # don't also count it as a manual flip
                print("[match] started")

            def lever(value):
                return 0 if value != value else int(value)  # NaN before first notification

            inputs = Inputs(
                match_on=match_on,
                out=ring.out,
                on_tape=ring.on_tape,
                switch=switch,
                button_tap=button_tap,
                flipper_ready=flipper.ready(),
                lever_left=lever(pad.sensor.leftPercent),
                lever_right=lever(pad.sensor.rightPercent),
            )

            # --- decide -------------------------------------------------------
            d = decide(inputs)

            # --- act ----------------------------------------------------------
            if d.fire:
                flipper.fire()

            drive = (d.left, d.right)
            if drive != last_drive or now - last_drive_sent >= DRIVE_HEARTBEAT:
                link.drive(*drive)
                last_drive, last_drive_sent = drive, now

            if d.state != last_state:
                print(f"[state] {last_state} -> {d.state}")
                link.state({"state": d.state, "color": color, "switch": switch})
                if d.state == "LOST":
                    link.drive(0, 0)
                    flipper.stop()
                    threading.Thread(target=play_sad_song, args=(motor,), daemon=True).start()
                    print("[match] pushed out. Press the controller button to re-arm.")
                last_state = d.state

            if now - last_print >= 0.5:
                print(f"  {d.state:<11} levers {inputs.lever_left:4d} {inputs.lever_right:4d}"
                      f"  drive {d.left:4d} {d.right:4d}  color {color}"
                      f"  switch {'ON ' if switch else 'off'}{'' if switch_fresh else ' (stale)'}")
                last_print = now

            time.sleep(max(0.0, LOOP_PERIOD - (time.monotonic() - now)))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        link.close()
        flipper.stop()
        for device in (motor, sensor, pad):
            try:
                device.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
