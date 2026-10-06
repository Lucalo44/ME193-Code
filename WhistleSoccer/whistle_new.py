"""Whistle-controlled Single Motor arm for the ME193 World Cup.

A second player runs this on their own laptop to control a Single Motor
arm on the same robot, while whistle_soccer.py drives the car. The arm
sweeps slowly back and forth across the top half of its circle only:
between the positive x-axis (horizontal right) and the negative x-axis
(horizontal left), passing through straight up -- never below horizontal.

Controls (whistle inside a band in FREQ_BANDS below):
  SWEEP   -- hold the note and the arm moves in its current direction. When
    it reaches a limit it turns around by itself and heads back, so a long
    SWEEP whistle swings the arm back and forth.
  REVERSE -- flip direction right away, then keep moving while held.
  silence -- the arm stops and holds its position.

The arm uses go-to-position commands aimed at the limits, so the motor
itself stops at the limit instead of relying on this script to notice in
time -- it never swings past horizontal even if the laptop lags.

MQTT: listens on the same topic as whistle_soccer.py. The arm stays still
until MSG_START arrives, and stops for good once the match ends
(MSG_FAILED or MSG_GOAL). It never publishes anything.

Setup:
  AirPods mic -- same as whistle_control.py: select AirPods as the input
    device in System Settings > Sound > Input, and grant microphone access
    to whatever runs this script under System Settings > Privacy &
    Security > Microphone.
  Single Motor -- update CARD_COLOR/CARD_SERIAL to match its Connection
    Card. Only one computer can connect to it at a time.

Calibrating (the arm must know where the x-axis is):
  1. Point the arm along the positive x-axis (horizontal, to the right) by
     hand and run the script -- it prints the motor's absolute position at
     startup, and the status box shows it live. Put that number in
     POSITIVE_X_ABS_POSITION.
  2. Whistle SWEEP. If the arm swings down under the x-axis instead of up
     over the top, flip UP_DIRECTION between 1 and -1.
  If the arm is geared, the 180-degree sweep is in motor degrees, not arm
  degrees -- scale SWEEP_DEGREES by the gear ratio.

Press 'q', Ctrl+C, or close the window to quit.
"""

import math
import signal
import sys
import time

import legoeducation as le
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.transforms import blended_transform_factory

from mqttlib import MQTTClient

# --- MQTT -------------------------------------------------------------------
MQTT_TOPIC = "ME193/Rogers"
# Must match the strings used by whistle_soccer.py and your opponent.
MSG_START = "start"
MSG_FAILED = "failed"
MSG_GOAL = "goal"

# --- Hardware -----------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_GREEN  # placeholder - replace with your Single Motor's actual card
CARD_SERIAL = "0994"               # placeholder - replace with your Single Motor's actual card

ARM_SPEED = 15             # percent -- keep low so the arm moves slowly
POSITIVE_X_ABS_POSITION = 0  # motor absolute position (0-359) when the arm points
                             # along the positive x-axis -- see "Calibrating" above
UP_DIRECTION = 1           # +1 or -1: which way the motor turns to go from +x up
                           # over the top to -x. Flip it if the arm swings underneath.
SWEEP_DEGREES = 180        # +x axis to -x axis
ARRIVE_TOLERANCE = 3       # within this many degrees of a limit counts as
                           # "arrived", which turns the arm around
COMMAND_HOLD_FRAMES = 3    # a new command must be heard this many frames in a row
                           # (~70 ms) before it's sent, so a wobbly whistle
                           # flickering across a band edge doesn't spam the motor
KEEPALIVE_SECONDS = 5      # LEGO devices power themselves off after ~1 minute of
                           # hearing nothing from the laptop -- even while connected.
                           # Since motor commands are only sent on change, poke every
                           # connected device this often so it stays on.
NOTIFICATION_DELAY_MS = 100  # sensor/motor update interval -- the keep-alive re-sends
                             # this same value, so it changes nothing on the device

# --- Audio ----------------------------------------------------------------
CHUNK = 1024              # samples read per frame -- lower is more responsive,
                           # higher is more frequency resolution and less CPU load
FREQ_MIN = 500             # ignore energy below this -- below typical whistle range
FREQ_MAX = 5500            # ignore energy above this -- above typical whistle range
AMPLITUDE_THRESHOLD = 20000  # FFT magnitude below this counts as "not whistling".
                           # Retune while watching the spectrogram -- and set it high
                           # enough that your teammate's whistle for the car, picked
                           # up from across the table, doesn't move the arm.

# Frequency bands, low Hz, high Hz, command name, plot color. Must not
# overlap. Any whistle below 2500 Hz sweeps; 2500-3500 Hz reverses.
FREQ_BANDS = [
    (FREQ_MIN, 2500, "SWEEP", "tab:green"),
    (2500, 3500, "REVERSE", "tab:orange"),
]

# --- Spectrogram display --------------------------------------------------
HISTORY_COLUMNS = 200   # how many past frames the scrolling spectrogram shows
DB_FLOOR = 20            # magnitudes below this are drawn as the darkest color


def find_input_device(pa: pyaudio.PyAudio) -> int:
    """Return the AirPods input device index if one is found, else the
    system default input device."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0 and "airpods" in info.get("name", "").lower():
            return i
    return pa.get_default_input_device_info()["index"]


def command_for_frequency(freq: float, amplitude: float) -> str | None:
    if amplitude < AMPLITUDE_THRESHOLD:
        return None
    for low, high, name, _color in FREQ_BANDS:
        if low <= freq <= high:
            return name
    return None


def keep_alive(devices):
    """Re-send each device its notification interval (the value it already
    has), non-blocking. Harmless, but counts as activity, so the device's
    ~1-minute idle power-off never kicks in."""
    for device in devices:
        device.device_notification_request(NOTIFICATION_DELAY_MS, blocking=False)


def try_connect(motor) -> bool:
    try:
        motor.connect(
            card_color=CARD_COLOR, card_serial=CARD_SERIAL,
            device_notification_delay=NOTIFICATION_DELAY_MS,
        )
    except Exception as exc:
        print(f"Could not connect to the Single Motor: {exc}")
        return False
    # connect() returns silently (without raising) if no matching device was
    # found, rather than raising -- .connected is the only reliable signal.
    if not motor.connected:
        print("Could not find a Single Motor matching that Connection Card. Is it switched on? LEGO devices turn\n"
              "  themselves off after about a minute with no activity -- press its button to wake it,\n"
              "  and make sure no other laptop or the LEGO website is connected to it.")
        return False
    return True


def wrap_degrees(angle: float) -> float:
    """Wrap an angle into -180..180."""
    return (angle + 180) % 360 - 180


def find_sweep_limits(motor) -> tuple[int, int]:
    """Return the relative-encoder positions of the positive x-axis and the
    negative x-axis, with the path between them running over the top.

    The relative position counts continuously (no wrap at 360), so a move
    between the two limits always takes the path through the top and can't
    go the long way round underneath. The absolute position anchors them to
    the real arm angle no matter where the arm sat at startup.
    """
    absolute = motor.motor.absolutePosition
    if absolute > 360:  # a negative angle that arrived as an unsigned 16-bit value
        absolute -= 65536
    # Straight up is 90 degrees past +x in the UP_DIRECTION sense.
    center_abs = POSITIVE_X_ABS_POSITION + UP_DIRECTION * SWEEP_DEGREES / 2
    offset = wrap_degrees(absolute - center_abs)
    center = motor.motor.position - offset
    half = UP_DIRECTION * SWEEP_DEGREES / 2
    return int(round(center - half)), int(round(center + half))


def main():
    game_state = {"started": False, "ended": False}

    def on_mqtt_message(_topic, payload):
        payload = payload.strip()
        if payload == MSG_START:
            game_state["started"] = True
        elif payload in (MSG_FAILED, MSG_GOAL):
            game_state["ended"] = True

    mqtt_client = MQTTClient()
    mqtt_client.subscribe(MQTT_TOPIC, on_mqtt_message)
    print(f"Subscribed to {MQTT_TOPIC!r}, waiting for {MSG_START!r}...")

    pa = pyaudio.PyAudio()
    device_index = find_input_device(pa)
    device_info = pa.get_device_info_by_index(device_index)
    rate = int(device_info["defaultSampleRate"])
    print(f"Using input device: {device_info['name']!r} @ {rate} Hz")
    if "airpods" not in device_info["name"].lower():
        print(
            "This doesn't look like an AirPods mic -- check System Settings > "
            "Sound > Input and select your AirPods there."
        )

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )

    arm = le.SingleMotor()
    print("Connecting to Single Motor...")
    connected = try_connect(arm)
    pos_x_limit, neg_x_limit = -90, 90
    if connected:
        # Hold position whenever the arm stops, so gravity can't swing it
        # past horizontal. Blocking is fine here: it's a one-off, before
        # the loop starts.
        arm.motor_set_end_state(le.MOTOR_END_STATE_HOLD)
        pos_x_limit, neg_x_limit = find_sweep_limits(arm)
        print(
            f"Arm absolute position: {arm.motor.absolutePosition} "
            f"(POSITIVE_X_ABS_POSITION = {POSITIVE_X_ABS_POSITION})"
        )
    else:
        print("Running in mic/spectrogram preview-only mode (no motor commands will be sent).")

    freqs = np.fft.rfftfreq(CHUNK, d=1.0 / rate)
    band_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    plot_freqs = freqs[band_mask]
    window = np.hanning(CHUNK)

    spec_buffer = np.full((plot_freqs.size, HISTORY_COLUMNS), DB_FLOOR, dtype=float)

    plt.ion()
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [30, 1], "wspace": 0.08},
    )
    fig.subplots_adjust(right=0.78, left=0.1)
    fig.canvas.manager.set_window_title("Whistle Arm (q to quit)")

    im = ax.imshow(
        spec_buffer,
        aspect="auto",
        origin="lower",
        extent=[0, HISTORY_COLUMNS, plot_freqs[0], plot_freqs[-1]],
        cmap="inferno",
        vmin=DB_FLOOR,
        vmax=DB_FLOOR + 60,
    )
    ax.set_xlabel("time ->")
    ax.set_ylabel("frequency (Hz)")
    fig.colorbar(im, cax=cax, label="magnitude (dB)")

    label_transform = blended_transform_factory(fig.transFigure, ax.transData)
    for low, high, name, color in FREQ_BANDS:
        ax.axhspan(low, high, color=color, alpha=0.15)
        ax.text(
            0.86, (low + high) / 2, name, transform=label_transform,
            va="center", ha="left", color=color, fontweight="bold", clip_on=False,
        )

    pitch_marker = ax.axhline(
        plot_freqs[0], color="#39FF14", linewidth=2.2, alpha=0.0,
        path_effects=[pe.Stroke(linewidth=4.2, foreground="black"), pe.Normal()],
    )
    status_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=13, fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.5, pad=6),
    )
    arm_text = ax.text(
        0.02, 0.88, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=11, bbox=dict(facecolor="black", alpha=0.5, pad=5),
    )

    closed = False

    def on_close(_event):
        nonlocal closed
        closed = True

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_close(event) if event.key == "q" else None,
    )
    # The GUI event loop can swallow the default KeyboardInterrupt, so make
    # Ctrl+C just flag the loop to exit and fall through to the cleanup.
    signal.signal(signal.SIGINT, lambda *_: on_close(None))

    candidate = None
    candidate_frames = 0
    active_command = None  # debounced command currently in effect
    heading_to_neg_x = True  # which x-axis limit the arm is heading for
    sent_target = None     # limit the motor was last told to go to, or None if stopped
    last_keepalive = time.monotonic()

    try:
        while not closed and not game_state["ended"]:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            spectrum = np.abs(np.fft.rfft(samples * window))[band_mask]

            spec_buffer = np.roll(spec_buffer, -1, axis=1)
            spec_buffer[:, -1] = np.maximum(20 * np.log10(spectrum + 1e-6), DB_FLOOR)
            im.set_data(spec_buffer)

            peak_idx = int(np.argmax(spectrum))
            peak_freq = plot_freqs[peak_idx]
            peak_amplitude = spectrum[peak_idx]

            command = command_for_frequency(peak_freq, peak_amplitude) if game_state["started"] else None
            if command == candidate:
                candidate_frames += 1
            else:
                candidate, candidate_frames = command, 1
            if candidate_frames >= COMMAND_HOLD_FRAMES and candidate != active_command:
                if candidate == "REVERSE":
                    heading_to_neg_x = not heading_to_neg_x
                active_command = candidate

            if connected and not arm.connected:
                print("Single Motor disconnected -- continuing in preview-only mode.")
                connected = False
            if connected and time.monotonic() - last_keepalive >= KEEPALIVE_SECONDS:
                keep_alive([arm])
                last_keepalive = time.monotonic()

            position = arm.motor.position if connected else float("nan")
            # Degrees from the +x axis toward -x: 0 = +x, 90 = up, 180 = -x.
            angle = (position - pos_x_limit) * UP_DIRECTION

            # Only talk to the motor when something changes, and never
            # block: see the freeze fix in whistle_control.py.
            if connected and not math.isnan(position):
                if active_command in ("SWEEP", "REVERSE"):
                    target = neg_x_limit if heading_to_neg_x else pos_x_limit
                    if sent_target == target and abs(position - target) <= ARRIVE_TOLERANCE:
                        heading_to_neg_x = not heading_to_neg_x  # reached an x-axis -- turn around
                        target = neg_x_limit if heading_to_neg_x else pos_x_limit
                    if target != sent_target:
                        arm.motor_run_to_relative_position(target, speed=ARM_SPEED, blocking=False)
                        sent_target = target
                elif sent_target is not None:
                    arm.motor_stop(blocking=False)
                    sent_target = None

            if peak_amplitude >= AMPLITUDE_THRESHOLD:
                pitch_marker.set_ydata([peak_freq, peak_freq])
                pitch_marker.set_alpha(0.9)
            else:
                pitch_marker.set_alpha(0.0)

            label = active_command or "-"
            status_text.set_text(f"{peak_freq:.0f} Hz  ->  {label}")
            band_colors = {name: color for _low, _high, name, color in FREQ_BANDS}
            status_text.set_color(band_colors.get(active_command, "white"))

            if not game_state["started"]:
                phase = f"waiting for {MSG_START!r}..."
            else:
                phase = "GO"
            if connected:
                heading = "toward -x" if heading_to_neg_x else "toward +x"
                arm_text.set_text(
                    f"{phase}   arm {angle:.0f} deg from +x, {heading}   "
                    f"(abs {arm.motor.absolutePosition})"
                )
            else:
                arm_text.set_text(f"{phase}   (no motor)")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    finally:
        if game_state["ended"]:
            print("Match over -- stopping the arm.")
        if connected and arm.connected:
            arm.motor_stop(blocking=False)
            arm.disconnect()
        mqtt_client.close()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
