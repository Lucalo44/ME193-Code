"""
config.py -- every tunable constant for the ping pong game lives here.

World coordinates (used by physics, AI and the browser renderer):
    meters, origin at the center of the table surface, y up,
    the PLAYER's end is at negative z, the COMPUTER's end at positive z.
    +x is the right-hand side of the screen when viewed from behind the player.

Values that come from calibration files (swing_calibration.json,
latency_calibration.json) override the defaults below at runtime.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Hardware: LEGO Education Double Motor used as the paddle
# --------------------------------------------------------------------------
# Connection Card filter, same idea as Pose Racecar/gesture_car_control.py.
# Run `python tools/find_motor.py` to see the color and serial of every motor nearby.
#   CARD_COLOR:  a color name -- "red", "yellow", "blue", "teal", "green", "purple",
#                "white", "magenta", "orange", "azure" (or "LEGO_COLOR_RED" etc.)
#   CARD_SERIAL: the number printed on the card, e.g. "0994"
# BOTH must match when both are set. Set either one to None to not filter on it
# (both None = connect to the first Double Motor found).
CARD_COLOR = "purple"
CARD_SERIAL = "0998"
IMU_NOTIFICATION_MS = 15          # 15 ms (~66 Hz) is the fastest the motor allows; default 100 is too slow
IMU_BUFFER_S = 2.0                # how much IMU history the paddle keeps in its ring buffer
HAPTICS_ENABLED = False           # short motor buzz on a successful hit
HAPTICS_PULSE_MS = 60
HAPTICS_SPEED = 40

# Raw-unit guesses used until calibration says otherwise (see hardware/swing.py docstring).
DEFAULT_ACCEL_RAW_PER_G = 1000.0  # replaced at zero() by the measured at-rest magnitude
DEFAULT_GYRO_RAW_PER_DPS = 1.0    # raw gyro counts per degree/second -- verify with tools/imu_logger.py
ANGLE_RAW_PER_DEG = 10.0          # raw yaw/pitch/roll counts per degree (decidegrees) -- verified Oct 7
# Live stroke (Wii-style): drawing the paddle back draws the on-screen paddle back, the
# forward swing sweeps it through the ball. Defaults from our recordings; calibrate_swing.py
# learns each player's own (left-handed players: the signs flip automatically).
LIVE_STROKE = True
STROKE_FH_BACK_DEG = 80.0         # on-screen yaw of a full forehand backswing
STROKE_BH_BACK_DEG = -65.0        # ... and of a full backhand backswing (opposite sign)
ORIENTATION_SOURCE = "onboard"    # on-screen paddle: "onboard" = the motor's own fused angles (stable),
                                  # "fusion" = our gyro+accel filter (drifts on sharp flicks; fallback)

# --------------------------------------------------------------------------
# AprilTags (start the game + pick ball speed)
# --------------------------------------------------------------------------
TAG_FAMILY = "36h11"
TAG_IDS = {0: "slow", 1: "medium", 2: "fast"}
TAG_CONFIRM_FRAMES = 8            # tag must be seen in this many ...
TAG_HISTORY_FRAMES = 10           # ... of the last this-many frames

# --------------------------------------------------------------------------
# Speeds (m/s)
# --------------------------------------------------------------------------
BALL_SPEED_SLOW = 5.5             # opponent's base shot speed per setting
BALL_SPEED_MEDIUM = 7.0
BALL_SPEED_FAST = 9.0
OPPONENT_SPEED_JITTER = 0.5       # +/- random variation around the base speed
OPPONENT_HARD_RETURN_BONUS = 0.8  # max extra speed when returning the player's hardest shots
RETURN_SPEED_MIN = 5.5            # player's return speed at strength 0
RETURN_SPEED_MAX = 10.0           # player's return speed at strength 1
RETURN_STRENGTH_CURVE = 0.8       # speed = lerp(min, max, strength ** curve)

SPEED_BY_SETTING = {"slow": BALL_SPEED_SLOW, "medium": BALL_SPEED_MEDIUM, "fast": BALL_SPEED_FAST}

# --------------------------------------------------------------------------
# Swing detection (units: linear acceleration in g, gyro in deg/s)
# --------------------------------------------------------------------------
SWING_GYRO_THRESHOLD = 150.0      # deg/s
SWING_ACCEL_THRESHOLD = 1.2       # g, gravity removed
SWING_COINCIDENCE_S = 0.10        # gyro and accel thresholds must both trip within this span
SWING_HYSTERESIS = 0.6            # swing ends once both signals fall below threshold * this
SWING_END_QUIET_S = 0.05          # ... and stay there this long
SWING_MAX_DURATION_S = 0.6
SWING_COOLDOWN_S = 0.25           # refractory period: one swing -> one event (short, so a too-early
                                  # swing can be followed by a second, real one)
SWING_RECOVERY_S = 0.65           # a detection this soon after a swing ...
SWING_RECOVERY_RATIO = 0.8        # ... and weaker than this fraction of it is the return to ready, not a swing
SWING_STRIKE_WINDOW_S = (0.15, 0.08)  # swing features are measured this long before / after peak acceleration
SWING_PAST_PEAK_RATIO = 0.6       # a swing is reported once acceleration falls below this fraction of its peak
# Hit-stop: a real swing is only reported ~0.1-0.25 s after the strike (plus Bluetooth
# delay). So when the ball reaches the paddle while a swing is under way, it's held there
# until the swing registers and then launched FROM THE PADDLE -- no rewind, no jump.
HIT_HOLD_MAX_S = 0.35             # longest the ball waits at the paddle for a swing to register
HIT_HOLD_WAIT_S = 0.12            # ... and how long it waits when no swing has started (yet)
HIT_HOLD_EASE_S = 0.01            # the ball eases to a stop at the paddle over ~this time constant
                                  # (a soft catch, not a dead freeze; it travels at most speed x this)
DEFAULT_STRENGTH_MIN_G = 1.5      # peak linear accel mapped to strength 0 (overridden by calibration)
DEFAULT_STRENGTH_MAX_G = 6.0      # ... and to strength 1
SWING_CALIBRATION_FILE = "swing_calibration.json"

# Spin applied to the player's return at |topspin| = 1 / |sidespin| = 1
SPIN_MAX_RADS = 150.0             # topspin/backspin angular velocity, rad/s
SIDESPIN_MAX_RADS = 80.0

# --------------------------------------------------------------------------
# Pose / stroke check
# --------------------------------------------------------------------------
HANDEDNESS = "right"              # "right" or "left"
POSE_CLASSES = ["ready", "forehand", "backhand"]
POSE_K = 5
MIN_SAMPLES_PER_CLASS = 20
POSE_DATA_FILE = "pose_data.npz"
POSE_MODEL_PATH = os.path.join(HERE, "models", "pose_landmarker_lite.task")
POSE_HISTORY_S = 1.0              # ring buffer of per-frame predictions
POSE_FRAME_LATENCY_S = 0.05       # camera exposure -> frame arrival; subtracted from frame timestamps
STROKE_WINDOW_S = 0.15            # look at predictions within +/- this of the swing peak
CENTER_DEADBAND = 0.05            # m; either stroke is accepted this close to the centerline
SHOW_STROKE_HINT = True
WRONG_STROKE_PENALTY = "point"    # "point" or "game"
BURST_RECORD_INTERVAL_S = 0.1     # train_pose.py burst mode sample interval

# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
# Hit window, Wii-style: a swing that comes too EARLY is just a whiff -- swing again. Only
# LATE (or no swing) misses. The window is a reach: the ball can be struck from when it's
# HIT_REACH_EARLY_M before the paddle until HIT_REACH_LATE_M past it, so the time window is
# reach / ball speed -- a faster incoming ball gives a tighter window. Per difficulty.
HIT_REACH_EARLY_M = {"slow": 1.0, "medium": 0.9, "fast": 0.8}
HIT_REACH_LATE_M = {"slow": 0.8, "medium": 0.7, "fast": 0.6}
HIT_WINDOW_MIN_S = 0.07           # never tighter than this (either side)
HIT_WINDOW_MAX_S = 0.35           # never looser than this (either side)
LATE_ZONE_S = 0.3                 # swings up to this long after the window closes count as "LATE"
LATENCY_OFFSET_S = 0.0            # IMU/BLE/display latency; overridden by latency_calibration.json
LATENCY_CALIBRATION_FILE = "latency_calibration.json"
LATENCY_BEATS = 10
LATENCY_BEAT_PERIOD_S = 0.9
PLAYER_HIT_PLANE_Z = -1.55        # where the player meets the ball (just behind the end line)
OPPONENT_HIT_PLANE_Z = 1.55
SERVE_DELAY_S = 1.2               # pause before the CPU serves
POINT_OVER_S = 1.5
GAME_BREAK_S = 3.0                # pause between games of a match
LOOP_HZ = 120                     # game loop rate
PHYSICS_HZ = 240                  # physics sub-step rate
STATE_BROADCAST_HZ = 60
CAMERA_PREVIEW_FPS = 12

# --------------------------------------------------------------------------
# Table / ball geometry
# --------------------------------------------------------------------------
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_HEIGHT = 0.1525
NET_OVERHANG = 0.1525             # net posts stick out this far past the table sides
BALL_RADIUS = 0.02
FLOOR_Y = -0.76                   # table top is y = 0

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
GRAVITY = 9.81
DRAG_COEFF = 0.08                 # a_drag = -DRAG_COEFF * |v| * v           (1/m)
MAGNUS_COEFF = 0.004              # a_magnus = MAGNUS_COEFF * (spin x v)       (m/rad)
SPIN_DECAY_PER_S = 0.3            # spin *= exp(-SPIN_DECAY_PER_S * dt) in flight
RESTITUTION = 0.9                 # vertical coefficient of restitution on the table
TABLE_FRICTION_KEEP = 0.96        # horizontal speed kept on a bounce (before spin kick)
SPIN_BOUNCE_COEFF = 0.5           # how hard spin kicks the ball forward/back on a bounce
SPIN_BOUNCE_KEEP = 0.6            # spin kept after a bounce
NET_FACE_RESTITUTION = 0.15
NET_MARGIN = 0.03                 # aim solver keeps at least this much clearance over the net
ASSIST_LEVEL = 0.6                # 0..1, nudges player returns back onto the table
PLAYER_TARGET_Z = 0.95            # nominal depth the player's returns aim for (opponent half)
OPPONENT_TARGET_Z = -0.85         # nominal depth the opponent aims for (player half)
RALLY_TIMEOUT_S = 4.0             # a ball that hasn't been resolved in this long is dead

# --------------------------------------------------------------------------
# Opponent AI
# --------------------------------------------------------------------------
OPPONENT_MAX_SPEED = 3.0          # m/s lateral
OPPONENT_REACH = 0.55             # m; further than this from the ball = can't return it
OPPONENT_REACTION_S = {"slow": 0.25, "medium": 0.18, "fast": 0.12}
OPPONENT_MISS_BASE = 0.08
OPPONENT_MISS_PER_SPEED = 0.04    # added per m/s of incoming speed above OPPONENT_MISS_SPEED_REF
OPPONENT_MISS_SPEED_REF = 7.0
OPPONENT_MISS_PER_SPIN = 0.25     # added at full spin (|spin| = SPIN_MAX_RADS)
OPPONENT_MISS_MAX = 0.8
OPPONENT_WIDE_SHOT_CHANCE = 0.15  # occasional shots aimed close to the sideline
OPPONENT_SPIN_MAX = 0.5           # opponent spin as a fraction of SPIN_MAX_RADS

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
POINTS_TO_WIN = 11
GAMES_PER_MATCH = 3
SERVES_PER_TURN = 5               # the server switches every this-many points
DEUCE_SERVE_EVERY_POINT = False   # True = real-table-tennis rule: alternate every point at 10-10
PLAYER_SERVE = True               # False = the CPU serves every point (the old v1 behavior)

# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------
SERVE_POS = (0.25, 0.18, -1.6)    # where the player holds the ball (x, height above table, z)
SERVE_CONTACT_Y = 0.30            # the toss is struck as it falls back to this height
TOSS_HEIGHT_MIN = 0.22            # m; toss height for the gentlest flick ...
TOSS_HEIGHT_MAX = 0.55            # ... and the strongest
SERVE_ARM_DELAY_S = 0.6          # the toss is accepted this long after your serve comes up
TOSS_IGNORE_S = 0.15              # swings this soon after the toss are the toss itself, not a hit
SERVE_SPEED_MIN = 3.6             # player's serve speed at swing strength 0 (m/s)
SERVE_SPEED_MAX = 6.0             # ... and at strength 1
SERVE_SPIN_SCALE = 0.8            # serves carry this fraction of the swing's spin
SERVE_OWN_BOUNCE_Z = 0.7          # nominal depth of the serve's first bounce (own half, m from net)
CPU_SERVE_SPEED_FACTOR = 0.75     # CPU serve speed = base shot speed * this
# Toss detection on the real paddle: a sharp UPWARD jolt with little rotation.
TOSS_ACCEL_G = 0.8                # upward linear acceleration that counts as a toss (g)
TOSS_MAX_GYRO_DPS = 120.0         # more rotation than this = a swing, not a toss
TOSS_UP_FRACTION = 0.7            # the jolt must be mostly along "up"
TOSS_COOLDOWN_S = 1.0

# --------------------------------------------------------------------------
# Score publishing (MQTT): record number of continuous hits, as a float
# --------------------------------------------------------------------------
MQTT_ENABLED = True
MQTT_BROKER = "test.mosquitto.org"    # shared class broker (same as 9_22/mqttlib.py)
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Rogers/Luca"
RECORD_FILE = "streak_record.json"    # the record survives restarts

# --------------------------------------------------------------------------
# Server / UI
# --------------------------------------------------------------------------
HTTP_PORT = 8000
WS_PORT = 8765
OPEN_BROWSER = True


def _load_latency_override() -> None:
    global LATENCY_OFFSET_S
    path = os.path.join(HERE, LATENCY_CALIBRATION_FILE)
    try:
        with open(path) as f:
            LATENCY_OFFSET_S = float(json.load(f)["latency_offset_s"])
    except (OSError, ValueError, KeyError):
        pass


_load_latency_override()
