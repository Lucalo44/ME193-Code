# Navigation

A campus navigation device for the UNO Q that uses no GPS or Wi-Fi positioning. It's built on our
StepNet step counter (`GaitID/`, `ArduinoUNOq/gaitid/`) and the MPU-6050. The navigator itself is
a behavior-cloned policy, trained on demonstration walks from both partners plus DAgger corrections
(10 full walks max, including corrections).

## How it fits together

```
MPU-6050 (50 Hz, sketch) ──Bridge──> imu.py ──> steps.py (StepNet) ──> odometry.py
                                                 stairs.py (optional) ─┘   │
                                                                           v
                    web UI (phone) <──> app/python/main.py <── features.py (one observation per step)
                    record / navigate / DAgger                       │
                                                                     v
                                                         policy.py (BC model) ──> action + LED arrow
```

The policy sees step-level features (steps walked, heading change, recent turns, stairs, progress
along the route), never raw IMU. Raw IMU is too noisy to clone from directly.

## Layout

| Path | What it's for |
|---|---|
| `config.py` | Shared constants: sample rate, action set, routes, feature window |
| `nav/imu.py` | Parse IMU samples drained from the sketch; gyro-integrated heading |
| `nav/steps.py` | Wraps the numpy StepNet so each detected step becomes an event |
| `nav/stairs.py` | Optional stairs / flat classifier on per-step features |
| `nav/odometry.py` | Step-based dead reckoning: step count, heading, turn events |
| `nav/features.py` | Builds the per-step observation vector the policy sees |
| `nav/actions.py` | Action vocabulary (straight / left / right / stairs / arrive) |
| `nav/dataset.py` | Walk logs on disk (JSONL), demo vs DAgger, merging both partners |
| `nav/policy.py` | BC policy: numpy inference (board) + PyTorch training (laptop) |
| `app/` | The UNO Q App Lab app: sketch (IMU + LED arrows) + Python + phone web UI |
| `tools/sync_app.py` | Copies `nav/`, `config.py`, StepNet and models into `app/python/` |
| `tools/pull_walks.py` | Downloads walk logs from the board |
| `tools/clean_walks.py` | Review and fix action labels on a recorded walk before training |
| `tools/train.py` | Trains the policy on all demos + DAgger corrections |
| `tools/evaluate.py` | Leave-one-walk-out evaluation and offline replay |
| `data/walks/<person>/` | Recorded walks (one JSONL per walk) |
| `models/` | Trained policy weights |
| `tests/` | Feature and policy consistency checks |

## Workflow (planned)

1. **Demo:** pick a route on the phone page, walk it, tap the action as you take it.
2. **Clean:** fix late or missed taps in `tools/clean_walks.py`.
3. **Train:** `python3 tools/train.py` on both partners' walks.
4. **DAgger:** walk in navigate mode; when the device suggests the wrong action, tap the right one.
   The corrections are logged as new labeled data.
5. **Retrain** with the corrections, staying within the 10-walk budget.
