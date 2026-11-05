"""Every tunable number for the hand-lock lives here, so collect/train/run always agree."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"            # data/<task>/<label>/<session>_<n>.jpg
MODEL_DIR = HERE / "models"         # models/<task>.pt
PASSWORD_FILE = HERE / "password.json"

# ---------------- camera ----------------
CAM_INDEX = 0                       # fallback only; camera_picker.py saves the real choice
CAMERA_FILE = HERE / "camera.json"  # written by camera_picker.py
FRAME_W, FRAME_H = 640, 480
MIRROR = True                       # flip left/right; applied identically in collect + run

# Square "place your hand here" box, in fractions of the frame
ROI_CENTER = (0.5, 0.5)             # (x, y)
ROI_SIZE = 0.85                     # side length as a fraction of the frame height
SAVE_SIZE = 224                     # raw crops are saved at this size; nets resize from it

# ---------------- networks ----------------
CLASSES = {
    "gesture": ["none", "1", "2", "3", "4", "5"],   # "none" = empty box / hand leaving
    "owner": ["not_me", "me"],
}
GESTURE_SIZE = 96                   # gesture net input (RGB)
PALM_SIZE = 160                     # owner net input (CLAHE gray + crease map)
CREASE_KERNEL = 9                   # black-hat kernel (px at PALM_SIZE); ~2x a crease's width

# ---------------- lock behavior ----------------
CODE_LENGTH = 3
CONF_THRESH = 0.80                  # gesture softmax needed for a frame to count
HOLD_S = 3.0                        # hold a digit in the box this long to confirm it
GRACE_S = 0.35                      # dropouts/misreads shorter than this don't reset the hold
RELEASE_S = 0.3                     # box must read "none" this long before the next digit (allows repeats)
CONFIRM_FLASH_S = 0.6               # confirmed digit flashes green on the dial this long
OWNER_THRESH = 0.85                 # mean P(me) over a digit's frames needed to pass
ENTRY_TIMEOUT_S = 30                # time to enter all three digits after the button press
UNLOCK_HOLD_S = 6                   # door stays open this long, then relocks
DENIED_SHOW_S = 2
MAX_FAILS = 3                       # this many wrong attempts in a row...
LOCKOUT_S = 30                      # ...locks the keypad for this long
