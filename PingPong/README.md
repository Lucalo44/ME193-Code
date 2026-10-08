# PingPong — virtual ping pong with a LEGO Double Motor paddle

Hold a **LEGO Education Double Motor** as a paddle and play ping pong against
a computer opponent rendered in 3D in your browser.

- **AprilTags** start the game and pick the ball speed (tag 0 slow, 1 medium, 2 fast).
- The motor's **IMU** detects your swing. Swing harder for a faster return. Close or open
  the paddle face for topspin or backspin.
- A **pose classifier** checks the stroke. A ball arriving right of the screen centerline
  needs a forehand; one on the left needs a backhand (right-handed player). The wrong
  stroke loses the point.
- Python owns all game logic and physics. The browser (three.js) only renders.

Built from `PING_PONG_SPEC.md`, reusing patterns from `Pose Racecar/gesture_car_control.py`.

---

## Setup (macOS)

```bash
cd PingPong
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python tools/make_tags.py          # writes printable tags to tags/ (already generated)
```

The first run that uses the camera downloads the ~6 MB MediaPipe pose model into
`models/` (one-time, needs internet). After that, everything runs offline: three.js is
vendored in `web/vendor/`.

macOS will ask for **camera** and **Bluetooth** permission for your terminal app the first
time. Allow both.

### Slow first start (iCloud Drive)

If `~/Documents` syncs with iCloud Drive and "Optimize Mac Storage" is on, macOS offloads
rarely used files to the cloud. That includes the hundreds of library files in `venv`. The
camera then takes minutes to start, because MediaPipe has to download them one by one;
the CAMERA chip shows `loading Ns` until then. Keep the environment outside iCloud and
link it in:

```bash
python3.12 -m venv ~/.venvs/pingpong
~/.venvs/pingpong/bin/pip install -r requirements.txt
mv venv venv.icloud-old            # delete it once the new one works
ln -s ~/.venvs/pingpong venv       # venv/bin/python and `source venv/bin/activate` still work
```

The first run after this still takes ~20 s while macOS checks the new library files.
After that, the camera is ready in about 2 s.

## Run it

```bash
source venv/bin/activate

python main.py --no-motor --no-camera   # no hardware at all: keyboard paddle, keys 1/2/3 to start
python main.py --no-motor               # webcam (tags + pose), keyboard paddle
python main.py                          # full game: webcam + Double Motor
```

`main.py` serves the game at **http://localhost:8000** and opens your browser. Click into
the browser window so it gets your key presses. Stop the game with Ctrl-C in the terminal.
Shutdown stops the motor, disconnects Bluetooth and releases the camera.

Other flags:

| Flag | What it does |
|---|---|
| `--camera 1` | Pick a different webcam |
| `--pose-data FILE` / `--calibration FILE` | Use non-default pose data / swing calibration |
| `--use-default-calibration` | Play with the real paddle before running `calibrate_swing.py` |
| `--debug` | Debug overlay on; logs events to the terminal |
| `--seed N` | Reproducible rallies |
| `--no-browser` | Don't auto-open the browser |
| `--replay CSV` | Play a recorded IMU log (e.g. `imu_logs/calib_hard.csv`) as the paddle, in real time and looping. Lets you test without the motor. |
| `--record-file PATH` | Save the hit record somewhere other than `streak_record.json` (useful for test runs) |

With a real paddle, the game refuses to start until `swing_calibration.json` exists
(see Milestone 2) unless you pass `--use-default-calibration`.

## Score publishing (MQTT)

The game publishes your **record number of continuous hits** to the class broker:

| | |
|---|---|
| Broker | `test.mosquitto.org:1883` (same as `9_22/mqttlib.py`) |
| Topic | `ME193/Rogers/Luca` |
| Payload | the record as a floating-point number, e.g. `7.0` |

A continuous hit is one of your returns that lands legally on the opponent's side. The
streak carries on through points you win and breaks only when you lose a point. The
record is your best streak ever.

- **When it's sent:** once at startup, then every time you set a new record. The message
  is retained, so a late subscriber gets the current record right away.
- **Where it's saved:** `streak_record.json`, so it survives restarts.
- **On screen:** the scorebug shows `STREAK n / BEST n`, and the status list shows the
  MQTT connection.
- **No network:** the game keeps running, and the latest record is sent once the broker
  is reachable.

To watch it from another terminal:

```bash
mosquitto_sub -h test.mosquitto.org -t ME193/Rogers/Luca -v
```

| Flag | What it does |
|---|---|
| `--reset-record` | Start the record over at 0 (and publish `0.0`) |
| `--no-mqtt` | Don't publish |
| `--mqtt-topic T` | Publish to a different topic |

The broker, port, topic and file name are `MQTT_*` / `RECORD_FILE` in `config.py`.

## How to play

1. **Start:** in the lobby, hold tag 0, 1 or 2 up to the webcam until its ring fills
   (or press 1/2/3). The tag sets the opponent's shot speed for the whole match.
2. **Hold the paddle still at your ready position** when it connects. It zeroes itself
   then. Press **Z** any time to re-zero.
3. **Serving switches every 5 points**, like a real match (the yellow dot in the scorebug
   shows who serves). The opponent serves first.
   - **Your serve:** the ball waits in your hand and **YOUR SERVE** appears. Toss it by
     flicking the paddle sharply **straight up** (Space on the keyboard paddle; Shift+Space
     tosses higher). Then strike it out of the air with a normal swing as it falls through
     the ring.
   - Like a real serve, it must bounce on your half first, then the opponent's. The game
     aims it for you, and swing strength and face angle still set speed and spin. A
     serve that clips the net and lands is a **LET** and is replayed.
   - A swing while the ball is still rising is just a whiff, so swing again as it falls.
     Swinging too late is a fault, and so is letting the ball drop without swinging
     (**MISSED SERVE**).
4. When the opponent hits, watch where the ball is heading:
   - **Right of the centerline → forehand. Left → backhand.** (Inverted for left-handers: pick **LEFT-HANDED** under the speed cards in the lobby, or on the
     pause screen. The choice is saved in `player_settings.json`. For left-handers it also flips
     the paddle arm the camera follows, the on-screen paddle and the default backswing direction.)
     Close to the line, either stroke works.
   - The hint arrow and the colored half of the table show which stroke is needed
     (**H** toggles the hint).
5. **Swing when the ball reaches the ring.** The ring turns green while the ball can be
   struck.
   - **Too early is just a whiff**, like Wii games: "TOO EARLY — SWING AGAIN" shows, and
     you can swing again if you're quick. Only late or missing swings lose the point.
   - **The window depends on the ball and the difficulty.** You can strike the ball from
     when it's a set reach in front of you until a little past you, so faster balls give a
     tighter window. Typical windows:

     | Difficulty | Window around the ball's arrival |
     |---|---|
     | Slow | −270 / +215 ms |
     | Medium | −175 / +135 ms |
     | Fast | −120 / +90 ms |
   - Swing harder for a faster return.
   - A closed face (tilted down) gives topspin: the ball dips, and on the bounce it kicks
     forward. Use it to keep hard shots on the table.
   - An open face gives backspin: the ball floats and checks up. Hard flat or backspin
     shots can sail long.
   - Fast, spinny shots make the opponent miss more often.
6. **You lose the point** for **NO SWING**, **LATE**, **WRONG STROKE**, a return
   that goes **OUT** or into the **NET**, or a **SERVE FAULT** / **MISSED SERVE** on your serve.
7. Games go to 11 (win by 2) and a match is best of 3. On the game-over screen, show a
   tag to play again.

### Keys (browser window, or type into the terminal + Enter)

| Key | Action |
|---|---|
| `P` | Pause |
| `R` | Reset to lobby |
| `Z` | Re-zero paddle |
| `G` (or `D` with a real paddle) | Debug overlay |
| `1` `2` `3` | Start slow / medium / fast without a tag |
| `L` | Latency calibration (from the lobby) |
| `H` | Stroke hint on/off |
| `V` | Bloom on/off |
| `M` | Mute |
| `O` | Free orbit camera (debug) |

Keyboard paddle (`--no-motor`):

| Key | Action |
|---|---|
| Space | Swing, or toss the ball when it's your serve |
| Shift+Space | Hard swing, or a higher toss |
| hold `W` / `S` | Topspin / backspin |
| hold `A` / `D` | Sidespin |
| hold `F` / `B` | Force forehand / backhand. Only with `--no-camera`, for testing WRONG STROKE. Without it, every stroke counts as correct when the camera is off. |

In `--no-motor` mode, D is sidespin, so the debug overlay is on **G**.

---

## Milestones and manual tests

### 1. Skeleton, physics, 3D view, keyboard paddle

```bash
python -m pytest -q                       # physics, rules, swing, stroke-side, game-loop tests
python main.py --no-motor --no-camera     # press 2, then Space as the ball reaches the ring
```

**Check:**
- Rallies work, the score updates, and misses are labeled.
- Holding W or S while swinging visibly changes the trajectory.
- G shows the predicted arrival point, the hit plane and the velocity arrow.

### 2. IMU bring-up and calibration

```bash
python tools/imu_logger.py --name rest_then_rotate
```

**Check the units.** Hold still for the first 2 s, then rotate the paddle exactly 90°
about one axis and hold. Close the plot. The summary gives you:
- accel counts per g
- gyro counts per deg/s
- whether the angles are in degrees or decidegrees
- the real notification interval (15 ms requested)

Put those numbers in `config.py` (`DEFAULT_GYRO_RAW_PER_DPS`, `ANGLE_RAW_PER_DEG`) and in
the "Measured on our hardware" block of the `hardware/swing.py` docstring.

```bash
python tools/calibrate_swing.py              # guided, about 9 minutes (steps below)
python tools/calibrate_swing.py --swings 10  # 10 of each instead of 5: more reliable spin/stroke detection
python -m pytest -q tests/test_swing.py
```

The steps, all holding the paddle exactly as you'll play:

1. Hold still at your ready position.
2. Slowly **tilt the face down** about 45° and hold. This measures the gyro scale and how
   the motor's axes map onto the on-screen paddle.
3. 5 serve **tosses**: sharp flicks straight up, as if tossing the ball. This learns how
   hard, how straight and how twist-free your flick is, then checks that none of your
   recorded swings would count as a toss.
4. 5 soft + 5 hard **forehands**, then 5 soft + 5 hard **backhands**.
5. 5 closed-face + 5 open-face forehands, then the same for **backhands**. A backhand
   flips the face, so each stroke gets its own topspin/backspin reading.
6. 5 swings brushing left, 5 brushing right.

Each swing and toss step asks for 5 repetitions by default. `--swings 10` (any number
from 3 to 30) asks for more. That makes the spin and forehand/backhand detection more
reliable, with little extra gain past about 10–15, when fatigue starts to change your
swing. Every detected swing and flick is used, even if you do more or fewer than asked.

**How it decides spin and stroke.** For each decision — forehand vs. backhand, topspin
vs. backspin on each stroke, left vs. right sidespin — the tool trains a small classifier
(logistic regression, `hardware/swing_model.py`) on your labelled swings. It also builds
the older single-reading rule, and scores both by **leave-one-out**: each swing is
predicted by a model trained without it. The game uses whichever scores better.

Before training, the tool cleans each set:
- the return to ready after a swing is ignored;
- duplicate and start-up detections are dropped;
- features are measured around the moment of the strike;
- the swings are finally re-detected with the game's own thresholds, so the classifiers
  learn from exactly what the game will see.

**Check the calibration.** The tool prints:
- which IMU features it picked for topspin (forehand and backhand separately), sidespin,
  and telling forehands from backhands, each with a separation score (aim for 1.0 or
  more);
- the spin your closed/open and left/right swings score, whose signs should come out
  right;
- how many soft forehands and backhands it reads as the right stroke.
- how many of your 5 flicks count as tosses, and how many swings do (should be 0).
- the leave-one-out accuracy of each decision, classifier vs. single-reading rule, and
  which one the game will use.

**If a toss still isn't recognized in the game**, the game says why while you're due to
serve (e.g. "Toss not counted: too much rotation (310 deg/s, max 180)"). The debug
overlay (**D**) shows the numbers for your last flick.

**Run it again** if you calibrated before backhands and the tilt step were added; older
files still work but lack them.

**Turn recordings into tests.** Recordings with exactly 5 detected swings are saved as
`imu_logs/calib_*_5swings.csv`. Copy them into `tests/data/` and the swing tests assert
5 events per file. To recompute without the hardware, run
`python tools/calibrate_swing.py --from-logs`.

**Check the on-screen paddle.** Run `python main.py --no-camera`. Like Wii Sports, the
game moves the paddle to the ball and you control the stroke:
- As a ball comes in, the paddle **glides to the contact point** (the ring).
- With the real paddle the stroke is **live**:
  - **Draw back** (turn the paddle back) and the screen paddle draws back with you.
  - **Swing forward** and it sweeps through the ball into the follow-through at your
    speed.
  - Forehand or backhand is told by which way you draw back. A quick turn back through
    ready is a swing, a slow one is a new backswing.
  - How far *you* draw back is learned from your calibration swings (`hardware/stroke.py`).
  - The screen paddle mirrors your paddle's **full orientation**: any tilt, and a full
    360° turn. Turning past your full backswing eases the paddle back round through
    center at 180°, so a complete spin moves it smoothly with no jump between sides.
- With the keyboard paddle the stroke is scripted: an automatic wind-up, then Space swings
  through the ball.
- **Hit-stop.** A real swing only registers about 0.1 s after the strike, more with
  Bluetooth delay. So when the ball reaches your paddle mid-swing, the ball **and the
  screen paddle** stop together at the contact point. When the swing registers, the ball
  leaves from the paddle face. Before this it flew past you and reappeared up to ~0.7 m
  down the table.
- **Backswings aren't swings.** A burst of motion while you're drawing the paddle back
  is ignored; only the forward stroke counts.
- **Soft catch.** During the hit-stop the ball eases into the paddle over ~10 ms
  (`HIT_HOLD_EASE_S`) instead of freezing dead, and the screen paddle stays with it.
- **No mid-play computation hitches.** The computer's serve is planned while it's still
  holding the ball, and a legal serve per speed is found once at startup. The serve used
  to freeze the game for 60–90 ms on fast; the worst game-loop step is now ~17 ms, which
  the browser's 50 ms interpolation hides.
- **Smoothing.** The live stroke uses an adaptive (one-euro) filter: steady when you
  move slowly, so Bluetooth bursts don't make the paddle jitter, and almost no lag during
  a fast swing.
  - It waits at most `HIT_HOLD_MAX_S` (0.35 s), or `HIT_HOLD_WAIT_S` (0.12 s) if no swing
    has started. A miss just carries on.
- With the keyboard paddle, W/S/A/D tilt the face a little.

Press **Z** at your ready position to re-zero.

### 3. Real paddle in the game

```bash
python main.py --no-camera
```

**Check:**
- Each swing registers exactly once (the paddle flashes and the swing trace appears in
  the LAST SHOT card).
- Soft and hard swings give clearly different speeds.
- Closed and open faces give clearly different spin.

Tune these in `config.py` if needed:
- `RETURN_SPEED_MIN/MAX`
- `SPIN_MAX_RADS`, `MAGNUS_COEFF`
- `ASSIST_LEVEL` (0 = no help, 1 = maximum help)
- `SWING_COOLDOWN_S`

### 4. AprilTags

Print `tags/start_tags.pdf` (or the PNGs) with each tag at least 8 cm wide, keeping the
white border. Run `python main.py --no-motor`.

**Check:**
- Holding a tag fills its ring in the lobby, starts the game, and shows the tag outlined in
  the camera preview.
- A tag held up mid-rally does nothing.
- On the game-over screen, a tag starts a new match.

Tags are detected on the **unflipped** frame. Mirrored tags don't decode (verified).

### 5. Pose training and stroke check

```bash
python tools/train_pose.py                 # 0 ready, 1 forehand, 2 backhand; b = burst; s = save
python tools/train_pose.py --with-paddle   # y = swing-sync: real swings label the frames at the swing peak
python main.py
```

Record at least 20 samples per class (wind-up and follow-through under the same label).
Record plenty of "ready" too: standing, walking, holding the paddle.

**How the call is made:** WRONG STROKE is only given when the evidence is sure. The
camera's vote around the swing (it needs ≥ `STROKE_CAMERA_SURE` = 75% of the votes) is
combined with the paddle's own forehand/backhand guess from the swing calibration (≥
`STROKE_IMU_SURE`). If neither is sure, the camera saw only "ready" or no pose, or the two
confidently disagree, the swing counts as the right stroke. The message says which one
called it ("camera saw forehand" or "paddle saw forehand").

**Check:**
- The pose chip shows the live prediction.
- The wrong stroke shows **WRONG STROKE — needed BACKHAND** (or FOREHAND).
- `python -m pytest -q tests/test_stroke_side.py` passes.

Burst mode is a toggle (press 0/1/2 to start, press again or Space to stop). The spec
asked for hold-to-record, but OpenCV windows can't reliably tell a held key from macOS
key-repeat. Swing-sync mode arms a class with 0/1/2, and then each real swing labels it.

### 5b. Arm tracking (wrist/elbow drive the on-screen paddle)

The camera, AprilTags and MediaPipe now run in their own process (`vision/process.py`,
`VISION_PROCESS = True`), so pose work never stalls the game loop or Bluetooth. Every frame
also sends your shoulder/elbow/wrist positions. The game moves the on-screen paddle with
your hand and draws a faint forearm. The IMU still sets the paddle angle and the swing.

Arm tracking needs **no training**. Its only setup is your ready position:

```bash
python tools/arm_check.py                  # camera only: skeleton, paddle arm, screen-paddle dot
python tools/arm_check.py --with-paddle    # real swings confirm which arm holds the paddle
```

Stand where you'll play, hold the paddle at your ready position and press **z**.
- The **orange** arm should be the one holding the paddle. If it's the other arm, set
  your hand in the game's lobby (arm_check: press **h**). With `--with-paddle`, after 3 swings
  `[swings]` shows that the swings picked the arm.
- Move your hand. The red dot in the "screen paddle" panel should follow it smoothly.

In the game, press **Z** at your ready position. It zeroes the paddle and the arm
together. The game also re-zeroes the arm on its own when you're idle at serve time. The
**ARM** chip in the HUD shows the tracking state.

Settings in config.py: `POSE_ARM` (on/off), `POSE_ARM_WEIGHT` (how much of the hand's
motion reaches the paddle), `POSE_ARM_SCALE_M` (metres per torso length) and
`POSE_ARM_MAX_M` (clamp). If the camera process gives you trouble, set
`VISION_PROCESS = False` to go back to the in-game thread.

### 6. Latency calibration and polish

**Check:**
- From the lobby, press **L** and swing on each of the 10 pulses. The median offset
  between beat and swing peak is saved to `latency_calibration.json` and loaded
  automatically next time.
- The game runs at about 60 fps (shown in the debug overlay).
- Bloom (V) looks right on your machine. It starts off because it tinted the scene green
  in a software-rendered browser.
- Sound plays after your first key press. Browsers block audio until then.

---

## Hardware notes

- **Units.** The IMU's raw units aren't documented in `legoeducation` 1.1.1, so nothing
  depends on guessing them:
  - Accel scale is measured every time the paddle is zeroed (1 g at rest).
  - Gyro scale is measured by the tilt step of `calibrate_swing.py`. Angle units can be
    checked with `imu_logger.py`.
- **How you hold it matters.** Holding the motor bare in your hand versus mounted on a
  LEGO handle changes which axes mean "face angle", "brushing direction" and
  "forehand vs. backhand", and how the motor maps onto the screen. The calibration tool
  measures all of them from your swings. Recalibrate whenever the mounting changes.
- **On-screen orientation** comes from the motor's own fused yaw/pitch/roll, measured to
  be Z-Y-X Euler angles in decidegrees. They're converted to a quaternion relative to the
  zeroed ready pose, then mapped onto the screen using the calibration's tilt step.
  - On real recordings this stayed within about 1° of gravity, even after sharp serve
    flicks. Our own gyro integration drifted 40–50° there, so it's only a fallback:
    `ORIENTATION_SOURCE = "fusion"` in `config.py`.
- **Sample timing.** Bluetooth delivers IMU samples in bursts (half arrive under 5 ms
  apart, with gaps up to 300 ms), although the motor measures every ~15 ms.
  `SampleClock` re-times each sample onto the motor's even clock. That makes swing
  timing and anything integrated over time more accurate.
- **Notification rate.** Swings need 15 ms IMU notifications. `paddle.py` connects with
  `device_notification_delay=15` and records every sample through the notification
  callback (it doesn't poll `motor.imu_device`).
- **Motors.** They are never driven during play. Set `HAPTICS_ENABLED = True` for a short
  buzz on each hit.
- **Card filter.** `CARD_COLOR` / `CARD_SERIAL` in `config.py` pick your Double Motor by
  its Connection Card, like in Pose Racecar. Set either to `None` to skip that filter.

## Configuration

Every tunable is in `config.py`, with a comment for each.

| Setting | Values |
|---|---|
| `HANDEDNESS` | `"right"` or `"left"`: only the default. The lobby/pause switch overrides it (saved in `player_settings.json`) |
| `WRONG_STROKE_PENALTY` | `"point"` (default) or `"game"`, which ends the current game as a loss |
| `GAMES_PER_MATCH` | Number of games in a match |
| `ASSIST_LEVEL` | 0–1; how much returns get nudged back onto the table |
| `BALL_SPEED_*` | the opponent's shot speed for each tag |
| `HIT_REACH_EARLY_M`, `HIT_REACH_LATE_M` | Per difficulty: how far in front of / past you the ball can still be struck (the time window is reach ÷ ball speed). Early swings are whiffs; only late swings miss. |
| `OPPONENT_MISS_*` | Base miss chance, plus extra per unit of incoming speed and spin |
| `SERVES_PER_TURN` | Points per serve turn (5) |
| `DEUCE_SERVE_EVERY_POINT` | `True` = the standard rule of alternating every point at 10–10 |
| `PLAYER_SERVE` | `False` = the CPU serves every point |
| `TOSS_ACCEL_G`, `TOSS_MAX_GYRO_DPS`, `TOSS_UP_FRACTION` | Defaults for how sharp, how twist-free and how straight up a flick must be to count as a toss. `calibrate_swing.py` learns your own values and saves them in `swing_calibration.json`, which overrides these. |
| `TOSS_HEIGHT_*`, `SERVE_SPEED_*`, `SERVE_CONTACT_Y` | Toss height range, serve speed range, and the height the serve is struck at |

### Look and feel

The presentation is a tribute to Rockstar Games' *Table Tennis* (2006). All 3D models are
original, built from primitives; no Rockstar logos or fonts are used.

- A dark, sparsely designed venue with overhead lamps throwing a pool of light on a
  matte blue table, a red court mat, printed barriers, and a silhouette crowd.
- A different opponent for each speed, named after characters from that game:

  | Speed | Opponent | Look and movement |
  |---|---|---|
  | Slow | **Cassidy** (5'7", the defender) | Navy and sky-blue kit, brown ponytail. Stands upright with calm footwork. |
  | Medium | **Liu Ping** (5'9", the attacker) | Charcoal and orange kit, short hair. Deep, aggressive crouch. |
  | Fast | **Kumi** (5'2", the speedster) | Purple and lime kit, jaw-length bob. Bouncy, quick feet. |

  Kits and hairstyles follow each player's style, not nationality. Each model is lathed and jointed, with a face, a polo collar and a detailed paddle. The
  opponent only changes how the match looks: difficulty still comes from the speed
  setting in `config.py`. The names belong to Rockstar, so swap them out in
  `OPPONENTS` in `web/scene/theme.js` before publishing anything.
- **The ball glows in the color of its spin**, using the Xbox 360 face-button colors that
  game used:

  | Spin | Color |
  |---|---|
  | Topspin | green |
  | Backspin | yellow |
  | Curving left | blue |
  | Curving right | red |

- A broadcast-style HUD:
  - scorebug, top left
  - hardware and training status, top right
  - LAST SHOT panel, bottom left: spin compass in those four colors, a segmented power
    meter in m/s, spin/side/stroke readouts and the swing trace
  - camera picture-in-picture, bottom right
- Understated effects: soft bounce marks, a white contact flash, wooden "pock" and
  celluloid "tock" sounds, and crowd applause when you win a point.

To restyle the scene, edit `web/scene/theme.js`. It holds every color, light, material,
the camera, and each opponent's name, build, outfit and movement style, and no game logic lives in `web/`.

## Layout

```
config.py                tunables
main.py                  entry point, CLI, threads, clean shutdown
game/      physics.py    ball flight, drag, Magnus, bounces, net, aim solver
           opponent.py   computer opponent AI
           rules.py      scoring, serve rotation, centerline rule
           state_machine.py   LOBBY / SERVE / RALLY / POINT_OVER / GAME_OVER (+ LATENCY_CAL)
           loop.py       fixed-timestep loop, referee, hit window, swing handling
hardware/  paddle.py     Double Motor connection, IMU callback, ring buffer
           swing.py      swing detector, calibration, spin features
           sim_paddle.py keyboard stand-in
vision/    camera.py     capture thread
           apriltags.py  detector + debounce
           pose.py       landmarker, features, KNN, StrokeJudge
           worker.py     vision thread + camera preview
server/    protocol.py   all message schemas (mirrored in web/main.js)
           ws_server.py  HTTP static server + WebSocket
web/                     three.js scene (scene/*.js), HUD, vendored three.js r169
tools/                   train_pose, arm_check, imu_logger, calibrate_swing, make_tags
tests/                   pytest suite (synthetic IMU recordings in tests/synth.py)
```

## What's been verified so far

Verified without hardware:
- The pytest suite passes.
- The keyboard-paddle game runs end to end over the real WebSocket server, including full
  rallies.
- The page renders in Chrome with no console errors.
- The tag pipeline was checked on generated tag images.

Not yet verified (needs the hardware, then the tuning in Milestones 2–3):
- The Double Motor IMU path: units, thresholds, spin axes.
- The pose classifier on a real player.
