# GaitID step counter

This extends GaitID from naming my activity to counting my steps. The ME193 prompt was to "extend your walker to now count your steps using a neural net and PyTorch" and to "use every sensor you can to make it as accurate as possible." To meet both halves, I paired the MPU-6050 that already rides on my right ankle with the IMU inside a LEGO Education Double Motor strapped to my left ankle. A small temporal convolutional network in PyTorch fuses the two streams and counts.

The classifier in the rest of GaitID is untouched. This folder only reads the same UDP stream from the UNO Q.

## Hardware

1. MPU-6050 on the right ankle, wired to the UNO Q header pins exactly as before (Wire, 0x68, ±8g, ±1000°/s, 50Hz). The UNO Q app has to be in record mode, with `STREAM_TO` set to the laptop's IP in [app/python/main.py](../app/python/main.py).
2. Double Motor on the left ankle, switched on, talking BLE straight to the laptop at ~66Hz (15ms notifications, the fastest it allows). If more than one motor is nearby, set `LEGO_CARD_COLOR` and `LEGO_CARD_SERIAL` in [config.py](config.py). `python ../PingPong/tools/find_motor.py` lists what's advertising.

Of note, one IMU per ankle is deliberate. An ankle sensor sees its own heel strike as a sharp spike but the other foot's only as a faint echo. With a sensor on each side, every step lands directly under one of them.

## How it works

1. [sources.py](sources.py) puts both sensors on the laptop's `time.monotonic()` clock. The UNO Q's rows carry board time, so I take the smallest arrival-minus-board-time as the offset. The Double Motor's BLE notifications arrive in bursts, so they get re-timed onto an even clock, the same trick PingPong uses.
2. [fuse.py](fuse.py) resamples both onto one 50Hz grid. Per sensor, the network gets the accelerometer and gyro axes, |a|, |gyro|, and a 1/0 "present" flag, plus pitch and roll from the motor (yaw is a heading with an arbitrary zero, so I left it out). Accelerometer and angle channels have their 1s running mean removed. That strips gravity and strap tilt, which change every session and carry no step information.
3. [labels.py](labels.py) builds the training targets from a single hand count per take. Hand-timestamping every heel strike was not something I wanted to do for 20 minutes of walking. Instead, the labeler finds impact peaks across both sensors and keeps the N most prominent, where N is the count I entered. Each kept step becomes a Gaussian bump with an area of exactly 1.
4. [net.py](net.py) is StepNet. A 1×1 conv feeds five residual blocks with dilated convolutions (dilation 1, 2, 4, 8, 16), then a 1×1 conv and a softplus. The receptive field is 125 samples, or 2.5s, which covers a couple of strides around every sample. The output is a step density for every 20ms sample. The count over any stretch of time is just the sum of that density, so there is no peak picking and no threshold.
5. [training.py](training.py) trains on random 8s crops. Each crop gets a random strap rotation up to 20°, ±10% gain, up to ±60ms of clock skew between the sensors, and sensor dropout (25% of crops lose the motor, 15% lose the MPU). The dropout is why one network still counts when either sensor drops out. The loss is density MSE plus a small penalty on each crop's count error.

### What went wrong on the way

The first version of the network learned a flat cadence instead of individual steps. Three separate bugs were stacked on top of each other.

1. The labels were wrong. At 50Hz a heel strike is only 1–2 samples wide, so peaks of raw |a| hit or miss depending on sampling phase. Smoothing the absolute deviation over 100ms turned every strike into one clean bump.
2. Per-axis normalization broke rotation. Nearly all of the leg swing sits on one gyro axis, so a 20° rotation dumped 85°/s onto an axis with a 2°/s spread, a lovely 40σ input. Each accel and gyro triplet now shares one scale.
3. Normalization layers erased amplitude. GroupNorm rescaled standing still until it looked like walking, and BatchNorm's training statistics didn't match whole-take inference. The network trains fine with neither.

## Using it

From `GaitID/`, with the venv active:

```bash
# try everything without hardware
python stepcount/tools/make_synthetic_steps.py
python stepcount/tools/train_steps.py --data stepcount/data/synthetic --out stepcount/models/stepnet_synthetic.pt

# record real takes (count steps out loud, type the count at the end)
python stepcount/tools/record_steps.py walk --seconds 60
python stepcount/tools/record_steps.py run --seconds 45
python stepcount/tools/record_steps.py still --seconds 30      # enter 0
python stepcount/tools/record_steps.py bike --seconds 45       # enter 0

# check a take's labels, then train and cross-validate
python stepcount/tools/plot_take.py stepcount/data/raw/walk_1
python stepcount/tools/train_steps.py

# count live
python stepcount/tools/live_steps.py
```

Training uses the Mac's GPU (MPS) automatically. On the CPU, PyTorch's dilated convolutions took 200ms per batch, while MPS took 11ms. A 40-epoch run with 4-fold cross-validation finishes in about 2.5 minutes.

[train_steps.py](tools/train_steps.py) holds out whole takes and compares StepNet against a classic peak counter that thresholds the impact signal in g. On 16 synthetic takes, StepNet was off by 12.8 of 1139 steps (1.1%) and the peak counter by 96 (8.4%). Nearly all of the peak counter's error was 95 phantom steps on one biking take. On new synthetic takes, StepNet came in at 1.4% with both sensors, 1.8% with the MPU only, and 2.4% with the motor only. Synthetic data only proves the code path. Real recordings will be noisier, and those numbers will come down a peg.

[live_steps.py](tools/live_steps.py) runs the net on the last 10s every 0.25s and commits each 20ms sample exactly once, 1.5s behind real time, so the running total never double counts. Replaying a 38-step synthetic walk over UDP gave 37.

## Laptop-less mode

To count a walk to class without the laptop, the UNO Q logs and counts on its own. Only the MPU-6050 is used here. The Double Motor stays a laptop-only sensor for training takes and demos, since its BLE link goes to the laptop.

1. [app/python/main.py](../app/python/main.py) has three switches. `LOG = True` writes every sample to `logs/session_NNNN_<date>.csv` on the board, plus a summary JSON rewritten every 10s. `COUNT_STEPS = True` runs StepNet on the board. `DISPLAY = "steps"` puts the running count on the LED matrix in a 4×6 font: up to 999, then `1k2` for 1,200.
2. The board can't run PyTorch, so [board.py](board.py) is the same network in plain numpy, and [export_board.py](tools/export_board.py) dumps the trained weights to a 285KB `.npz`. The numpy copy matches PyTorch to 1e-4. A 10s pass takes 62ms on the UNO Q.
3. [pull_logs.py](tools/pull_logs.py) downloads the sessions over USB or Wi-Fi, recounts them on the laptop, and prints a report. `--label session_0003 --steps 412` turns a walk I counted into an MPU-only training take.

The session number leads each file name because the board's clock can drift without Wi-Fi, while a counter can't. Pulling the plug loses at most 10s of the on-board summary. The CSV keeps every sample, so the laptop recount is always complete.
