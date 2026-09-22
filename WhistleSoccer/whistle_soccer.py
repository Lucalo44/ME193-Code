"""Whistle Soccer: a whistle-controlled LEGO car that plays "ball" or
"goalie" in the ME193 World Cup.

Both roles drive the same way -- whistle inside one of the steering bands
in FREQ_BANDS below to move, same as whistle_control.py -- but differ in how
the match ends:

  ball -- tries to drive into the goal. A Color Sensor mounted facing
    forward watches for the goalie closing in (rising reflected light --
    see REFLECTION_THRESHOLD): if that happens, the ball stops, publishes
    MSG_FAILED, and plays DEATH_SONG. To score, whistle the separate
    GOAL_BAND pitch (held for GOAL_HOLD_FRAMES frames, so it can't be
    triggered by accident while steering): the ball stops, publishes
    MSG_GOAL, and plays GOAL_SONG.
  goalie -- just drives (to block the ball) and listens on the same MQTT
    topic. On MSG_FAILED (the ball got blocked) it plays VICTORY_SONG; on
    MSG_GOAL (the ball scored) it plays DEATH_SONG.

Both roles wait for MSG_START on MQTT_TOPIC before driving at all.

Setup:
  Set ROLE below to "ball" or "goalie" for this robot.
  AirPods mic -- same as whistle_control.py: select AirPods as the input
    device in System Settings > Sound > Input, and grant microphone access
    to whatever runs this script under System Settings > Privacy &
    Security > Microphone.
  Double Motor -- update MOTOR_CARD_COLOR/MOTOR_CARD_SERIAL to match your
    car's Connection Card.
  Color Sensor (ball only) -- mount it facing forward, unobstructed, and
    update COLOR_CARD_COLOR/COLOR_CARD_SERIAL to match its Connection Card.
  MQTT messages (MSG_START/MSG_FAILED/MSG_GOAL below) and MQTT_TOPIC must
    be agreed on with your opponent's team before the match -- both sides
    need the exact same strings and topic.

Calibrating:
  FREQ_BANDS/GOAL_BAND: run the script (no hardware needs to be connected
    to see the spectrogram -- it runs in preview-only mode) and whistle
    across your range, watching where the bright line lands. Leave gaps
    between all five bands.
  REFLECTION_THRESHOLD: once the ball's Color Sensor is connected, watch
    the printed reflection values (or add your own printout) as the
    goalie approaches from your working distance, and set the threshold
    partway between "nothing nearby" and "goalie in blocking range."

Press 'q' or close the window to quit early (no MQTT message is sent).
"""

import sys
import time

import legoeducation as le
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.transforms import blended_transform_factory

from mqttlib import MQTTClient

# --- Role -----------------------------------------------------------------
ROLE = "ball"  # "ball" or "goalie" -- set this per robot before each match

# --- MQTT -------------------------------------------------------------------
MQTT_TOPIC = "ME193/Rogers"
# These exact strings must match what your opponent's script sends/expects.
MSG_START = "start"    # match begins
MSG_FAILED = "failed"  # sent by the ball when the goalie blocks it
MSG_GOAL = "goal"      # sent by the ball when it whistles its way into the goal

# --- Hardware -----------------------------------------------------------
MOTOR_CARD_COLOR = le.LEGO_COLOR_PURPLE  # placeholder - replace with your Double Motor's actual card
MOTOR_CARD_SERIAL = "5164"               # placeholder - replace with your Double Motor's actual card
COLOR_CARD_COLOR = le.LEGO_COLOR_PURPLE  # placeholder - replace with your Color Sensor's actual card
COLOR_CARD_SERIAL = "5164"               # placeholder - replace with your Color Sensor's actual card

MOTOR_SPEED = 60        # speed sent to the motors when a command is active, in percent
RIGHT_MOTOR_SIGN = -1   # the two motors are mirror-mounted, so equal signed
                         # speeds spin the car in place instead of driving it
                         # straight -- this inverts the right side to cancel
                         # that out. Flip to +1 if it makes things worse.
TURN_SPEED_SCALE = 0.6  # LEFT/RIGHT drive one side slower than the other
                         # rather than fully reversing it, so the car arcs
                         # instead of spinning in place. 0 = spin in place,
                         # 1 = same speed as straight (no turning effect).

REFLECTION_THRESHOLD = 60   # ball only -- sensor.reflection (0-100) above this
                             # counts as "something is close in front of the
                             # sensor." Placeholder -- tune against your own
                             # working distance and lighting.
REFLECTION_HOLD_READS = 3   # consecutive above-threshold sensor reads
                             # required before treating it as a real block,
                             # not one noisy reading.

# --- Audio ----------------------------------------------------------------
CHUNK = 1024              # samples read per frame -- lower is more responsive,
                           # higher is more frequency resolution and less CPU load
FREQ_MIN = 500             # ignore energy below this -- below typical whistle range,
                           # cuts out most voice/room noise
FREQ_MAX = 5000            # ignore energy above this -- above typical whistle range
AMPLITUDE_THRESHOLD = 20000  # FFT magnitude below this counts as "not whistling" ->
                           # stop. Starting point only -- measured against ambient
                           # room/laptop-fan noise on a built-in mic, which alone hit
                           # 3,000-19,000 in testing. AirPods have different mic
                           # gain, so retune this while watching the spectrogram:
                           # too low and background noise triggers commands, too
                           # high and quiet whistles get missed.

# Steering bands, low Hz, high Hz, command name, plot color. Must not
# overlap each other or GOAL_BAND; leave gaps for a reliable "no command" zone.
FREQ_BANDS = [
    (700, 1100, "FORWARD", "tab:green"),
    (1300, 1700, "BACKWARD", "tab:red"),
    (1900, 2300, "LEFT", "tab:blue"),
    (2500, 2900, "RIGHT", "tab:orange"),
]
# Ball only -- a deliberately separate, higher pitch so it's never confused
# with steering. Must be held for GOAL_HOLD_FRAMES consecutive frames.
GOAL_BAND = (3300, 3700, "GOAL", "gold")
GOAL_HOLD_FRAMES = 15  # ~0.5-0.75s at the audio loop's natural frame rate

# --- Songs ------------------------------------------------------------------
# User-defined: each note is (frequency_hz, duration_seconds). Frequency is
# capped at 2700 Hz by the hardware. Edit freely.
DEATH_SONG = [(400, 0.3), (350, 0.3), (300, 0.3), (200, 0.6)]
VICTORY_SONG = [(523, 0.15), (659, 0.15), (784, 0.15), (1047, 0.4)]
GOAL_SONG = [(784, 0.15), (880, 0.15), (988, 0.15), (1175, 0.15), (1568, 0.5)]

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


def is_goal_whistle(freq: float, amplitude: float) -> bool:
    if amplitude < AMPLITUDE_THRESHOLD:
        return False
    low, high, _name, _color = GOAL_BAND
    return low <= freq <= high


def try_connect(device, card_color, card_serial, label: str) -> bool:
    try:
        device.connect(card_color=card_color, card_serial=card_serial)
    except Exception as exc:
        print(f"Could not connect to the {label}: {exc}")
        return False
    # connect() returns silently (without raising) if no matching device was
    # found, rather than raising -- .connected is the only reliable signal.
    if not device.connected:
        print(f"Could not find a {label} matching that Connection Card.")
        return False
    return True


def drive(car, connected: bool, command: str | None):
    if not connected:
        return
    if command == "FORWARD":
        car.motor_run(motor=le.MOTOR_LEFT, speed=MOTOR_SPEED, blocking=False)
        car.motor_run(motor=le.MOTOR_RIGHT, speed=RIGHT_MOTOR_SIGN * MOTOR_SPEED, blocking=False)
    elif command == "BACKWARD":
        car.motor_run(motor=le.MOTOR_LEFT, speed=-MOTOR_SPEED, blocking=False)
        car.motor_run(motor=le.MOTOR_RIGHT, speed=-RIGHT_MOTOR_SIGN * MOTOR_SPEED, blocking=False)
    elif command == "LEFT":
        car.motor_run(motor=le.MOTOR_LEFT, speed=int(MOTOR_SPEED * TURN_SPEED_SCALE), blocking=False)
        car.motor_run(motor=le.MOTOR_RIGHT, speed=RIGHT_MOTOR_SIGN * MOTOR_SPEED, blocking=False)
    elif command == "RIGHT":
        car.motor_run(motor=le.MOTOR_LEFT, speed=MOTOR_SPEED, blocking=False)
        car.motor_run(motor=le.MOTOR_RIGHT, speed=RIGHT_MOTOR_SIGN * int(MOTOR_SPEED * TURN_SPEED_SCALE), blocking=False)
    else:
        car.motor_stop(motor=le.MOTOR_BOTH)


def play_song(device, song, connected: bool):
    """Play a (frequency_hz, duration_s) melody on the hub's speaker."""
    if not connected:
        print("(not connected -- would play a song here)")
        return
    for frequency, duration in song:
        device.beep(frequency=int(frequency), count=1, blocking=False)
        time.sleep(duration)
        device.stop_beep(blocking=True)
        time.sleep(0.03)  # brief gap between notes


def main():
    if ROLE not in ("ball", "goalie"):
        raise ValueError(f"ROLE must be 'ball' or 'goalie', got {ROLE!r}")

    game_state = {"started": False, "ended": False, "outcome": None}

    def on_mqtt_message(_topic, payload):
        payload = payload.strip()
        if payload == MSG_START:
            game_state["started"] = True
        elif ROLE == "goalie" and not game_state["ended"]:
            if payload == MSG_FAILED:
                game_state["ended"] = True
                game_state["outcome"] = "victory"
            elif payload == MSG_GOAL:
                game_state["ended"] = True
                game_state["outcome"] = "death"

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

    car = le.DoubleMotor()
    print("Connecting to Double Motor...")
    motor_connected = try_connect(car, MOTOR_CARD_COLOR, MOTOR_CARD_SERIAL, "Double Motor")
    if not motor_connected:
        print("Running without the Double Motor connected (no motor commands or songs will play).")

    color_sensor = None
    sensor_connected = False
    if ROLE == "ball":
        color_sensor = le.ColorSensor()
        print("Connecting to Color Sensor...")
        sensor_connected = try_connect(color_sensor, COLOR_CARD_COLOR, COLOR_CARD_SERIAL, "Color Sensor")
        if not sensor_connected:
            print("Running without the Color Sensor connected (blocking by the goalie won't be detected).")

    freqs = np.fft.rfftfreq(CHUNK, d=1.0 / rate)
    band_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    plot_freqs = freqs[band_mask]
    window = np.hanning(CHUNK)

    spec_buffer = np.full((plot_freqs.size, HISTORY_COLUMNS), DB_FLOOR, dtype=float)

    display_bands = FREQ_BANDS + ([GOAL_BAND] if ROLE == "ball" else [])

    plt.ion()
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [30, 1], "wspace": 0.08},
    )
    fig.subplots_adjust(right=0.78, left=0.1)
    fig.canvas.manager.set_window_title(f"Whistle Soccer -- {ROLE} (q to quit)")

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
    for low, high, name, color in display_bands:
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
    role_text = ax.text(
        0.02, 0.90, f"role: {ROLE}", transform=ax.transAxes, va="top", ha="left",
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

    goal_hold_counter = 0
    reflection_hold_counter = 0

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

            if not game_state["started"]:
                role_text.set_text(f"role: {ROLE}  --  waiting for {MSG_START!r}...")
                drive(car, motor_connected, None)
            else:
                role_text.set_text(f"role: {ROLE}  --  GO")
                command = command_for_frequency(peak_freq, peak_amplitude)
                drive(car, motor_connected, command)

                if ROLE == "ball" and sensor_connected:
                    if color_sensor.sensor.reflection >= REFLECTION_THRESHOLD:
                        reflection_hold_counter += 1
                    else:
                        reflection_hold_counter = 0
                    if reflection_hold_counter >= REFLECTION_HOLD_READS:
                        game_state["ended"] = True
                        game_state["outcome"] = "blocked"

                if ROLE == "ball" and not game_state["ended"]:
                    if is_goal_whistle(peak_freq, peak_amplitude):
                        goal_hold_counter += 1
                    else:
                        goal_hold_counter = 0
                    if goal_hold_counter >= GOAL_HOLD_FRAMES:
                        game_state["ended"] = True
                        game_state["outcome"] = "scored"

            if peak_amplitude >= AMPLITUDE_THRESHOLD:
                pitch_marker.set_ydata([peak_freq, peak_freq])
                pitch_marker.set_alpha(0.9)
            else:
                pitch_marker.set_alpha(0.0)

            command = command_for_frequency(peak_freq, peak_amplitude) if game_state["started"] else None
            label = command or "-"
            status_text.set_text(f"{peak_freq:.0f} Hz  ->  {label}")
            band_colors = {name: color for _low, _high, name, color in display_bands}
            status_text.set_color(band_colors.get(command, "white"))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        pass
    finally:
        drive(car, motor_connected, None)

        outcome = game_state["outcome"]
        if outcome == "blocked":
            print("Blocked by the goalie -- publishing failure and playing the death song.")
            mqtt_client.publish(MQTT_TOPIC, MSG_FAILED)
            play_song(car, DEATH_SONG, motor_connected)
        elif outcome == "scored":
            print("Goal! Publishing the score and playing the goal song.")
            mqtt_client.publish(MQTT_TOPIC, MSG_GOAL)
            play_song(car, GOAL_SONG, motor_connected)
        elif outcome == "victory":
            print("The ball was blocked -- playing the victory song.")
            play_song(car, VICTORY_SONG, motor_connected)
        elif outcome == "death":
            print("The ball scored -- playing the death song.")
            play_song(car, DEATH_SONG, motor_connected)

        if motor_connected:
            car.motor_stop(motor=le.MOTOR_BOTH)
            car.disconnect()
        if sensor_connected:
            color_sensor.disconnect()
        mqtt_client.close()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
