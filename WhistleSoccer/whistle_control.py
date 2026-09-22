"""Drive a LEGO Double Motor (car) by whistling into a pair of AirPods.

The car is controlled purely by pitch: whistle inside one of the frequency
bands defined in FREQ_BANDS below and the car drives that direction for as
long as you hold the note. Stop whistling (or land between bands) and it
stops. A live spectrogram shows exactly what the microphone is hearing, with
the command bands drawn on top, so you can see your whistle's actual pitch
against the zone boundaries in real time instead of guessing.

Setup:
  AirPods mic -- in System Settings > Sound > Input, select your AirPods as
    the input device (just being Bluetooth-connected isn't enough; macOS
    still defaults input to the built-in mic unless you pick AirPods
    explicitly). Also grant microphone access to whatever runs this script
    (Terminal/IDE) under System Settings > Privacy & Security > Microphone.
  Double Motor -- update CARD_COLOR/CARD_SERIAL below to match your car's
    Connection Card.

Calibrating FREQ_BANDS:
  Everyone's whistle range is different. Run the script (the car doesn't
  need to be connected to see the spectrogram -- it'll just run in
  preview-only mode), whistle low, then high, and watch where the bright
  band lands on the frequency axis. Edit FREQ_BANDS below so your
  comfortable low/high whistles land inside FORWARD/BACKWARD and your
  in-between pitches land inside LEFT/RIGHT (or however you want them
  mapped) -- leave gaps between bands so you can land on "no command"
  between them.

Press 'q' or close the window to quit.
"""

import sys
import time

import legoeducation as le
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.transforms import blended_transform_factory

# --- Hardware -----------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_PURPLE  # placeholder - replace with your Double Motor's actual card
CARD_SERIAL = "5164"               # placeholder - replace with your Double Motor's actual card

MOTOR_SPEED = 60        # speed sent to the motors when a command is active, in percent
RIGHT_MOTOR_SIGN = -1   # the two motors are mirror-mounted, so equal signed
                         # speeds spin the car in place instead of driving it
                         # straight -- this inverts the right side to cancel
                         # that out. Flip to +1 if it makes things worse.
TURN_SPEED_SCALE = 0.6  # LEFT/RIGHT drive one side slower than the other
                         # rather than fully reversing it, so the car arcs
                         # instead of spinning in place. 0 = spin in place,
                         # 1 = same speed as straight (no turning effect).

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

# Frequency bands, low Hz, high Hz, command name, plot color. Must not
# overlap; leave gaps between bands for a reliable "no command" zone.
FREQ_BANDS = [
    (700, 1100, "FORWARD", "tab:green"),
    (1300, 1700, "BACKWARD", "tab:red"),
    (1900, 2300, "LEFT", "tab:blue"),
    (2500, 2900, "RIGHT", "tab:orange"),
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


def try_connect(car) -> bool:
    try:
        car.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    except Exception as exc:
        print(f"Could not connect to the Double Motor: {exc}")
        return False
    # connect() returns silently (without raising) if no matching device was
    # found, rather than raising -- .connected is the only reliable signal.
    if not car.connected:
        print("Could not find a Double Motor matching that Connection Card.")
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


def main():
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
    connected = try_connect(car)
    if not connected:
        print("Running in mic/spectrogram preview-only mode (no motor commands will be sent).")

    freqs = np.fft.rfftfreq(CHUNK, d=1.0 / rate)
    band_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    plot_freqs = freqs[band_mask]
    window = np.hanning(CHUNK)

    spec_buffer = np.full((plot_freqs.size, HISTORY_COLUMNS), DB_FLOOR, dtype=float)

    plt.ion()
    # Explicit main/colorbar axes (rather than letting fig.colorbar() resize
    # ax automatically) so the layout is fixed up front -- the band-name
    # labels are placed in the reserved right margin and stay there
    # regardless of what colorbar/subplots_adjust do afterward.
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [30, 1], "wspace": 0.08},
    )
    fig.subplots_adjust(right=0.78, left=0.1)
    fig.canvas.manager.set_window_title("Whistle Control (q to quit)")

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

    # x in figure-fraction (just right of the subplots_adjust(right=0.78)
    # boundary, so it always lands in the reserved empty margin no matter
    # how ax/cax split the space to its left), y in ax's data coordinates.
    label_transform = blended_transform_factory(fig.transFigure, ax.transData)
    for low, high, name, color in FREQ_BANDS:
        ax.axhspan(low, high, color=color, alpha=0.15)
        ax.text(
            0.86, (low + high) / 2, name, transform=label_transform,
            va="center", ha="left", color=color, fontweight="bold", clip_on=False,
        )

    # Bright, black-outlined line so it stays visible against every part of
    # the "inferno" colormap, from near-black background to washed-out
    # yellow/white peaks.
    pitch_marker = ax.axhline(
        plot_freqs[0], color="#39FF14", linewidth=2.2, alpha=0.0,
        path_effects=[pe.Stroke(linewidth=4.2, foreground="black"), pe.Normal()],
    )
    status_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=13, fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.5, pad=6),
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

    try:
        while not closed:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            spectrum = np.abs(np.fft.rfft(samples * window))[band_mask]

            spec_buffer = np.roll(spec_buffer, -1, axis=1)
            spec_buffer[:, -1] = np.maximum(20 * np.log10(spectrum + 1e-6), DB_FLOOR)
            im.set_data(spec_buffer)

            peak_idx = int(np.argmax(spectrum))
            peak_freq = plot_freqs[peak_idx]
            peak_amplitude = spectrum[peak_idx]

            command = command_for_frequency(peak_freq, peak_amplitude)
            drive(car, connected, command)

            if peak_amplitude >= AMPLITUDE_THRESHOLD:
                pitch_marker.set_ydata([peak_freq, peak_freq])
                pitch_marker.set_alpha(0.9)
            else:
                pitch_marker.set_alpha(0.0)

            label = command or "-"
            status_text.set_text(f"{peak_freq:.0f} Hz  ->  {label}")
            status_text.set_color("white" if command is None else next(
                color for low, high, name, color in FREQ_BANDS if name == command
            ))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        pass
    finally:
        if connected:
            car.motor_stop(motor=le.MOTOR_BOTH)
            car.disconnect()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
