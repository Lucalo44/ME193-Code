# Campus Navigator

The assignment asks for a campus navigation device that uses no GPS and no Wi-Fi positioning, built on top of our step counters and trained by behavior cloning. Our route is the classroom to the Makerspace. We get at most 10 full walks for training, DAgger walks included, and the navigator has to learn from both partners' data.

The navigator works like a phone GPS. A new user straps the board to their right ankle, opens a page on their phone, and gets "In 30 m, turn left", then "Turn left now", then "You have arrived at the Makerspace". A second page on the laptop shows the raw IMU, the movement prediction (still, walking, running, stairs), and a live view of how the policy is making each decision.

## How it works

Everything runs on the UNO Q. The STM32 samples the ankle MPU-6050 at 50Hz (the same sketch format our StepNet was trained on), and the Linux side runs this chain on every sample:

```
raw IMU ─> attitude filter (heading) ─> StepNet (steps) ─> stairs + activity (per step)
        ─> odometry (turns, position) ─> 19 step-level features ─> policy ─> instruction
```

1. **Heading.** The MPU rides on the ankle, where the shank swings at 300-500°/s every stride. My first heading estimate projected the gyro onto a low-passed gravity vector. This was not a major success: the sway turned into a fake rotation of hundreds of degrees per minute. I replaced it with a small Mahony-style quaternion filter in [nav/imu.py](nav/imu.py). It only trusts the accelerometer during stance, when the foot is planted (the zero-velocity trick from foot-mounted navigation). It also learns the gyro bias whenever the board sits still, which took standing drift from 61° to 6° per minute on our data.
2. **Steps.** Our trained StepNet runs in plain numpy ([nav/stepcount_board.py](nav/stepcount_board.py)). [nav/steps.py](nav/steps.py) turns its density into one event per step. StepNet commits samples 1.5 s late, so steps arrive about 2 steps behind real time.
3. **Movement.** Still, walking and running come from our GaitID activity classifier ([nav/activity.py](nav/activity.py)). Stairs come from a per-step logistic regression on vertical acceleration and step timing ([nav/stairs.py](nav/stairs.py)). It's trained from the stairs taps in the walks.
4. **Odometry.** [nav/odometry.py](nav/odometry.py) keeps the step count, a dead-reckoned path, and the turns made so far (more than 50° over 6 steps).
5. **Features.** The policy never sees raw IMU, as the hints recommend. [nav/features.py](nav/features.py) builds one vector per step: progress, heading, recent heading change, left and right turn counts, steps since the last turn, stairs, movement, cadence and position.
6. **Policy.** [nav/policy.py](nav/policy.py) is a two-layer MLP with two heads: the next maneuver (left, right, stairs, arrive) and the number of steps until it. Those two numbers are exactly what a GPS voice needs. It trains in PyTorch on the laptop and runs in numpy on the board. For the dashboard, it also reports gradient × input for every feature, so you can see what pushed it toward "left".

The same [nav/pipeline.py](nav/pipeline.py) runs live on the board, in the laptop simulator, and in training (which replays each recorded walk through it). A feature in training is therefore exactly the feature on a walk.

## Labels, cleaning and DAgger

Labeling during a demonstration really is hard: taps land a second or two late, or get missed. Each walk stores the raw IMU and the taps, never features, so walks can be re-processed after any change. Before training, [nav/walks.py](nav/walks.py) snaps every turn tap to the actual turn in the heading data within 12 steps, and shifts other taps back by a reaction time. [tools/clean_walks.py](tools/clean_walks.py) draws the result (heading, taps, cleaned labels) and allows manual fixes before saving `labels.json`.

DAgger works through the phone page. In DAgger mode the policy navigates, and you tap the maneuvers you actually make. The states come from walking under the policy, the labels come from you, and those walks count double in training.

## Using it

Set up the laptop side once (from `Navigation/`):

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

1. **Deploy:** `python3 tools/sync_app.py --deploy` assembles the app in `ArduinoUNOq/navigation/` and pushes it.
2. **Record demos:** open `http://<board>:8080/` on a phone, choose your name, and press **Record demo**. Walk to the Makerspace, tapping Left, Right, Stairs start / Stairs end as you make each one, and **Arrived** at the end.
3. **Pull and clean:** `python3 tools/pull_walks.py`, then `python3 tools/clean_walks.py`. Check each `review.png`, fix anything, and run again with `--save`.
4. **Train:** `python3 tools/train.py --loo` trains on both partners' walks and prints leave-one-walk-out results. Then run `python3 tools/sync_app.py --restart-python`.
5. **DAgger:** walk in **DAgger walk** mode, pull, clean, retrain. Keep the total at 10 walks.
6. **Navigate:** **Navigate** on the phone. The dashboard is at `http://<board>:8080/dash` on the laptop.

To try it without hardware, `python3 tools/make_test_walks.py` builds synthetic routes from our real GaitID walking (straightened, with 90° turns injected and late taps). `python3 tools/train.py --test` trains on them. `python3 tools/sim.py --test` serves both pages at `localhost:8080` with the walk replayed in real time.

## Status

Everything runs end to end on synthetic walks. Label cleaning lands within 0-1 steps of the true turns, turn detection finds 24 of 24 turns with no false ones, and leave-one-walk-out maneuver accuracy is 98-100%. Synthetic walks repeat one template, so real walks will be harder. No real walks exist yet, and the stairs model stays off until there are stairs taps.

Two things to check on the first real walk:
- `STRIDE_M` in [config.py](config.py) only sets the metres shown on screen. Calibrate it from a walk of known length.
- The activity classifier was trained on ankle data. The board needs to sit in the same place and orientation.
