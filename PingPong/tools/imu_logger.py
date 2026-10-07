#!/usr/bin/env python3
"""
tools/imu_logger.py -- record raw Double Motor IMU notifications to CSV and
live-plot the six raw channels. Use it first (Milestone 2) to learn the units
and noise of the IMU before calibrating swings.

    python tools/imu_logger.py --name rest_then_rotate
    python tools/imu_logger.py --name forehand_soft_5swings --seconds 20 --no-plot

Writes imu_logs/<name>.csv (columns: t, yaw, pitch, roll, ax, ay, az, gx, gy, gz,
raw integers, t = time.monotonic()). Close the plot window or press Ctrl-C to
stop. On exit it prints a summary that answers the unit questions in the
hardware/swing.py docstring:

  * accel raw per g  -- hold the paddle still for the first 2 s; the mean
                        accel magnitude over that time is 1 g.
  * gyro raw per dps -- after the still period, rotate the paddle exactly 90
                        degrees about one axis and hold; the integrated gyro on
                        that axis divided by 90 is raw-counts per deg/s.
  * angle units      -- compare the change in pitch/roll/yaw to the 90 degrees
                        you rotated (900 = decidegrees).

Recordings named like *_5swings.csv can be copied into tests/data/ and are
then checked by tests/test_swing.py.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import config as C  # noqa: E402
from hardware.paddle import Paddle  # noqa: E402
from hardware.swing import Calibration, ImuSample, save_csv  # noqa: E402

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


def connect(paddle: Paddle, timeout: float = 30.0) -> bool:
    paddle.start()
    from hardware.paddle import describe_card_filter
    print(f"Connecting to {describe_card_filter()} (turn it on)...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if paddle.status == "connected":
            return True
        if paddle.status not in ("connecting", "disconnected"):
            break
        time.sleep(0.2)
    print(f"Paddle status: {paddle.status}")
    return False


def summarize(samples) -> None:
    if len(samples) < 10:
        print("Not enough samples for a summary.")
        return
    n = len(samples)
    dts = [b.t - a.t for a, b in zip(samples, samples[1:])]
    dts.sort()
    print(f"\n{n} samples over {samples[-1].t - samples[0].t:.1f} s; "
          f"median interval {dts[len(dts) // 2] * 1000:.1f} ms (requested {C.IMU_NOTIFICATION_MS} ms), "
          f"95th pct {dts[int(len(dts) * 0.95)] * 1000:.1f} ms")
    still = [s for s in samples if s.t - samples[0].t < 2.0]
    mags = [math.sqrt(s.ax ** 2 + s.ay ** 2 + s.az ** 2) for s in still]
    mean_mag = sum(mags) / len(mags)
    print(f"First 2 s (hold still!): accel magnitude {mean_mag:.1f} raw  -> accel raw per g = {mean_mag:.1f}")
    for ch in CHANNELS:
        vals = [getattr(s, ch) for s in still]
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        print(f"   {ch}: mean {mean:8.1f}  noise sd {sd:6.1f}")
    print("Integrated gyro after the still period (raw counts x seconds):")
    rest = [s for s in samples if s.t - samples[0].t >= 2.0]
    for ch in ("gx", "gy", "gz"):
        total = sum(getattr(a, ch) * (b.t - a.t) for a, b in zip(rest, rest[1:]))
        print(f"   {ch}: {total:9.1f}   (if you rotated 90 deg about this axis: gyro raw per dps = {abs(total) / 90:.2f})")
    print("Angle ranges (raw):")
    for ch in ("yaw", "pitch", "roll"):
        vals = [getattr(s, ch) for s in samples]
        print(f"   {ch}: {min(vals):8.0f} .. {max(vals):8.0f}   (a 90 deg tilt reading ~900 means decidegrees)")
    peak = max(math.sqrt(s.ax ** 2 + s.ay ** 2 + s.az ** 2) for s in samples)
    print(f"Peak accel magnitude {peak:.0f} raw = {peak / mean_mag:.1f} g"
          + ("   <-- near int16 limit, the sensor may be saturating" if peak > 30000 else ""))


def live_plot(samples, lock, stop: threading.Event, window_s: float = 5.0) -> None:
    import matplotlib.pyplot as plt

    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.canvas.manager.set_window_title("Double Motor IMU (close to stop)")
    lines = {}
    for ax_i, group in zip(axes, (CHANNELS[:3], CHANNELS[3:])):
        for ch in group:
            (lines[ch],) = ax_i.plot([], [], label=ch)
        ax_i.legend(loc="upper left")
        ax_i.grid(alpha=0.3)
    axes[0].set_ylabel("accel (raw)")
    axes[1].set_ylabel("gyro (raw)")
    axes[1].set_xlabel("seconds")
    while not stop.is_set() and plt.fignum_exists(fig.number):
        with lock:
            recent = list(samples[-int(window_s * 100):])
        if recent:
            t_end = recent[-1].t
            recent = [s for s in recent if t_end - s.t <= window_s]
            ts = [s.t - t_end for s in recent]
            for ch, line in lines.items():
                line.set_data(ts, [getattr(s, ch) for s in recent])
            for ax_i in axes:
                ax_i.relim()
                ax_i.autoscale_view()
            axes[1].set_xlim(-window_s, 0)
        plt.pause(0.05)
    stop.set()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=time.strftime("imu_%Y%m%d_%H%M%S"))
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until closed / Ctrl-C)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    paddle = Paddle(Calibration(), auto_zero=False)
    samples: list = []
    lock = threading.Lock()

    def on_sample(s: ImuSample) -> None:
        with lock:
            samples.append(s)

    paddle.raw_listeners.append(on_sample)
    if not connect(paddle):
        return 1
    print("Connected. Hold the paddle STILL for the first 2 seconds...")
    stop = threading.Event()
    t0 = time.time()
    try:
        if args.no_plot:
            while not stop.is_set():
                time.sleep(0.5)
                print(f"\r{len(samples)} samples", end="", flush=True)
                if args.seconds and time.time() - t0 > args.seconds:
                    break
        else:
            if args.seconds:
                threading.Timer(args.seconds, stop.set).start()
            live_plot(samples, lock, stop)
    except KeyboardInterrupt:
        pass
    finally:
        paddle.shutdown()
    path = os.path.join(HERE, "imu_logs", f"{args.name}.csv")
    with lock:
        save_csv(path, samples)
        summarize(samples)
    print(f"\nSaved {len(samples)} samples to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
