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
    "voice": ["none", "1", "2", "3", "4", "5"],     # spoken "one".."five"; none = noise/other words
    "speaker": ["not_me", "me"],
}
TASK_EXT = {"gesture": ".jpg", "owner": ".jpg", "voice": ".wav", "speaker": ".wav"}
AUDIO_TASKS = ("voice", "speaker")
GESTURE_SIZE = 96                   # gesture net input (RGB)
PALM_SIZE = 160                     # owner net input (CLAHE gray + crease map)
CREASE_KERNEL = 9                   # black-hat kernel (px at PALM_SIZE); ~2x a crease's width

# ---------------- voice ----------------
MIC_FILE = HERE / "mic.json"        # written by mic_picker.py
SAMPLE_RATE = 16000
CLIP_S = 1.0                        # every voice clip is 1 s
N_MELS = 40                         # mel bands -> net input is 40 x 98
WIN_MS, HOP_MS = 25, 10             # STFT window / hop
TRIGGER_DB = -35                    # lowest dBFS that starts a clip (adjust live with +/-); room was ~-45
TRIGGER_MARGIN_DB = 12              # ...and it must also be this far above the room's noise floor
MIC_GAIN_PCT = 50                   # UNO Q webcam mic capture volume (it boots at 100% = +24 dB)
PREROLL_S = 0.2                     # clip starts this long before the trigger
TRIGGER_COOLDOWN_S = 0.25           # ignore the tail of a word after a clip is cut
VOICE_CONF = 0.70                   # live_voice.py shows "?" below this
SPEAKER_THRESH = 0.85               # P(me) needed for the speaker check

# ---------------- lock behavior ----------------
CODE_LENGTH = 3
VOTE_CONF = 0.50                    # a frame votes for its top label when it's at least this sure
HOLD_S = 3.0                        # a digit needs this many seconds of votes to be confirmed
HOLD_DECAY = 0.5                    # a vote for something else drains other digits at this rate
GRACE_S = 0.8                       # no digit votes for this long = hand gone, hold resets
RELEASE_S = 0.5                     # no digit votes for this long before the next digit (allows repeats)
CONFIRM_FLASH_S = 0.6               # confirmed digit flashes green on the dial this long
OWNER_THRESH = 0.85                 # mean P(me) over a digit's frames needed to pass
ENTRY_TIMEOUT_S = 30                # time to enter all three digits after the button press
GRANTED_SHOW_S = 3                  # check mark on the matrix after a correct code, then the open padlock
DENIED_SHOW_S = 2
MAX_FAILS = 3                       # this many wrong attempts in a row...
LOCKOUT_S = 30                      # ...locks the keypad for this long
