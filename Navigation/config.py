"""Shared constants for the navigator. The laptop tools and the UNO Q app both import this."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "walks"          # data/walks/<person>/<walk id>/  (meta.json, imu.csv, taps.json)
TEST_DIR = HERE / "data" / "test_walks"     # synthetic walks from tools/make_test_walks.py (never trained on by default)
MODEL_DIR = HERE / "models"
WEB_DIR = HERE / "web"

# ---------------- route ----------------
ROUTE = "Classroom -> Makerspace"
START, GOAL = "Classroom", "Makerspace"

# ---------------- sensing ----------------
FS = 50                      # IMU sample rate (Hz); SAMPLE_HZ in the sketch must match
STEP_MODEL = "stepnet_board.npz"            # our trained StepNet (numpy, from GaitID/)
ACTIVITY_MODEL = "activity.json"            # GaitID model 2 (still / walk / run / bike)
STAIRS_MODEL = "stairs.json"                # trained by tools/train.py from stairs taps
POLICY_MODEL = "policy.npz"                 # the behavior-cloned navigator
STEP_UPDATE_S = 0.5          # run StepNet this often (it commits steps 1.5 s behind real time)
GRAVITY_TAU_S = 1.0          # low-pass time constant for the gravity direction (yaw axis)
STRIDE_M = 0.72              # metres per step, only for the "in 40 m" display

# ---------------- turn detection (odometry) ----------------
TURN_WINDOW = 6              # steps over which a heading change counts as one turn
TURN_DEG = 50                # heading change over TURN_WINDOW steps that makes a turn
TURN_REFRACTORY = 6          # steps after a turn before another can start

# ---------------- maneuvers ----------------
# What the navigator tells you to do next. "stairs" = take the stairs here.
MANEUVERS = ["left", "right", "stairs", "arrive"]
TAP_ACTIONS = ["left", "right", "stairs_start", "stairs_end", "arrive"]   # buttons in record mode
NOW_STEPS = 3                # "turn left NOW" when the predicted distance is this close
SNAP_STEPS = 12              # label cleaning: move a turn tap to a real turn within this many steps
REACTION_STEPS = 2           # taps with no turn nearby: assume the tap came this many steps late

# ---------------- training ----------------
MAX_WALKS = 10               # assignment cap: demos + DAgger walks combined
DAGGER_WEIGHT = 2.0          # DAgger-corrected walks count this much more in training
HIDDEN = 48                  # policy MLP width
PEOPLE = ["luca", "rafae"]   # both partners' walks train one navigator
