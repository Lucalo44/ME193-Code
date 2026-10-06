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

With a real paddle, the game refuses to start until `swing_calibration.json` exists
(see Milestone 2) unless you pass `--use-default-calibration`.

## How to play

1. **Start:** in the lobby, hold tag 0, 1 or 2 up to the webcam until its ring fills
   (or press 1/2/3). The tag sets the opponent's shot speed for the whole match.
2. **Hold the paddle still at your ready position** when it connects. It zeroes itself
   then. Press **Z** any time to re-zero.
3. The opponent serves every point. Watch where the ball is heading:
   - **Right of the centerline → forehand. Left → backhand.** (Inverted if `HANDEDNESS = "left"`.)
     Close to the line, either stroke works.
   - The hint arrow and the colored half of the table show which stroke is needed
     (**H** toggles the hint).
4. **Swing when the ball reaches the ring.** The ring turns green during the hit window
   (±0.2 s).
   - Swing harder for a faster return.
   - A closed face (tilted down) gives topspin: the ball dips, and on the bounce it kicks
     forward. Use it to keep hard shots on the table.
   - An open face gives backspin: the ball floats and checks up. Hard flat or backspin
     shots can sail long.
   - Fast, spinny shots make the opponent miss more often.
5. **You lose the point** for **NO SWING**, **EARLY**, **LATE**, **WRONG STROKE**, or a return
   that goes **OUT** or into the **NET**.
6. Games go to 11 (win by 2) and a match is best of 3. On the game-over screen, show a
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
| Space | Swing |
| Shift+Space | Hard swing |
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
python tools/calibrate_swing.py          # guided: still, soft/hard, closed/open face, brush left/right
python -m pytest -q tests/test_swing.py
```

**Check the calibration.** The tool prints which IMU features it picked for topspin and
sidespin, with a separation score (aim for 1.0 or more). It then re-scores your
closed/open and left/right swings, and their signs should come out right.

**Turn recordings into tests.** Recordings with exactly 5 detected swings are saved as
`imu_logs/calib_*_5swings.csv`. Copy them into `tests/data/` and the swing tests assert
5 events per file. To recompute without the hardware, run
`python tools/calibrate_swing.py --from-logs`.

**Check the ghost paddle.** Run `python main.py --no-camera`: the ghost paddle at the
bottom of the screen should tilt with the real paddle.

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

**Check:**
- The pose chip shows the live prediction.
- The wrong stroke shows **WRONG STROKE — needed BACKHAND** (or FOREHAND).
- `python -m pytest -q tests/test_stroke_side.py` passes.

Burst mode is a toggle (press 0/1/2 to start, press again or Space to stop). The spec
asked for hold-to-record, but OpenCV windows can't reliably tell a held key from macOS
key-repeat. Swing-sync mode arms a class with 0/1/2, and then each real swing labels it.

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
  - Gyro and angle scales are checked with `imu_logger.py`.
- **How you hold it matters.** Holding the motor bare in your hand versus mounted on a
  LEGO handle changes which axes mean "face angle" and "brushing direction". The
  calibration tool picks them from your swings. Recalibrate whenever the mounting changes.
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
| `HANDEDNESS` | `"right"` or `"left"` |
| `WRONG_STROKE_PENALTY` | `"point"` (default) or `"game"`, which ends the current game as a loss |
| `GAMES_PER_MATCH` | Number of games in a match |
| `ASSIST_LEVEL` | 0–1; how much returns get nudged back onto the table |
| `BALL_SPEED_*` | the opponent's shot speed for each tag |
| `HIT_WINDOW_S` | How far before or after the ball arrives a swing still counts |
| `OPPONENT_MISS_*` | Base miss chance, plus extra per unit of incoming speed and spin |

### Look and feel

The presentation is a tribute to Rockstar Games' *Table Tennis* (2006). All 3D models are
original, built from primitives; no Rockstar logos or fonts are used.

- A dark, sparsely designed venue with overhead lamps throwing a pool of light on a
  matte blue table, a red court mat, printed barriers, and a silhouette crowd.
- A different opponent for each speed, named after characters from that game:

  | Speed | Opponent | Look and movement |
  |---|---|---|
  | Slow | **Cassidy** (IRL, 5'7", the defender) | Green polo, white skirt, auburn ponytail. Stands upright with calm footwork. |
  | Medium | **Liu Ping** (CHN, 5'9", the attacker) | Red and gold polo, short black hair. Deep, aggressive crouch. |
  | Fast | **Kumi** (JPN, 5'2", the speedster) | White and red polo, navy shorts, hair buns. Bouncy, quick feet. |

  Each model is lathed and jointed, with a face, a polo collar and a detailed paddle. The
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
tools/                   train_pose, imu_logger, calibrate_swing, make_tags
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
