# GaitID

Wearable activity classifier on the Arduino UNO Q + a 6-axis IMU (assumed MPU-6050, "5060").
It tells apart **standing still, walking, running, and biking** (or lying on your back air-biking).

Two models, compared head to head:

| | Input | Classifier | File |
|---|---|---|---|
| **Model 1** | the peak acceleration in **one** direction (`max|a - median|` of one accel axis, per 2 s window) | Gaussian naive Bayes on 1 feature | `models/model1_peak.json` |
| **Model 2** | **all** IMU outputs: 6 channels' mean/std/peak, \|accel\| and \|gyro\| stats, dominant motion frequency | standardized logistic regression | `models/model2_full.json` |

Ankle placement is easiest (big, distinct motion); hip is harder (small, similar-looking motion).
Record both and compare with `tools/evaluate.py`; the interesting question is how much
Model 1 loses at the hip and how much Model 2 recovers.

## Layout

```
GaitID/
├── gaitid/          shared library: config, features, model (numpy-only inference), data, synth
├── app/             UNO Q App Lab app (deploys to the board)
│   ├── sketch/      STM32: samples IMU at 50 Hz, serves it over the Bridge, draws S/W/R/B on the matrix
│   └── python/      Linux: streams samples to laptop and/or runs the model live
├── tools/           record.py · train.py · evaluate.py · sync_app.py · make_synthetic.py · demo.py
├── data/raw/        your recordings:  <label>_<n>.csv   (label = still|walk|run|bike)
├── models/          trained model JSON + confusion matrices
├── tests/
└── venv/            (git-ignored)
```

## Setup

```bash
cd GaitID
source venv/bin/activate          # already created; to rebuild:
# python3.12 -m venv venv && pip install -r requirements.txt
pytest                            # sanity check
```

## Workflow

1. **Try the pipeline without hardware:** `python tools/make_synthetic.py && python tools/train.py --data data/synthetic`
2. **Wire it up:** IMU on the Qwiic connector (`Wire1`; edit `IMU_WIRE` in `app/sketch/sketch.ino` otherwise).
   Strap it to the ankle (or hip). Keep the same orientation every time.
3. **Collect data:** in `app/python/main.py` set `STREAM_TO` to your laptop's IP, then
   `python tools/sync_app.py && python3 ../ArduinoUNOq/tools/deploy.py gaitid` (from `GaitID/`, the sync tool prints the exact command).
   Then per take: `python tools/record.py walk --seconds 60`. Do **at least 3–4 separate takes per class**
   (cross-validation holds out whole recordings). Vary pace/speed within a class.
4. **Train & compare:** `python tools/train.py` prints grouped-CV accuracy and confusion matrices for both
   models and writes them to `models/`. Model 1's axis is chosen by CV (`--axis az` to force).
5. **Run live:** set `STREAM_TO = None`, `MODEL = "model1_peak.json"` or `"model2_full.json"`,
   `python tools/sync_app.py`, redeploy. The LED matrix shows S / W / R / B.
6. **Test generalization:** record a separate folder (e.g. `data/hip/`) and run `python tools/evaluate.py data/hip`.
7. **Group demo:** with the board in record mode (`STREAM_TO` set), `python tools/demo.py` opens a window
   with a stick figure that copies the detected activity and walks each groupmate through walk → run → bike
   (30 s each, then a hand-off screen; Space continues). `--sim` fakes the activity with the S/W/R/B keys.

## Notes

- Units: accel in g, gyro in deg/s, 50 Hz. Change `FS` in `gaitid/config.py` **and** `SAMPLE_HZ` in the sketch together.
- Biking vs. walking is where Model 1 should struggle: pedalling is smooth, so its peak overlaps still/walk.
- `app/python/gaitid/`, `app/python/models/` are copies made by `sync_app.py`; edit `gaitid/`, not the copies.
- Synthetic data only validates the code path, never report its accuracy.
