"""Shared constants. Copied to the board with the rest of the gaitid package."""

CLASSES = ["still", "walk", "run", "bike"]
LETTERS = {"still": "S", "walk": "W", "run": "R", "bike": "B"}   # shown on the LED matrix

FS = 50                  # sample rate in Hz (must match SAMPLE_HZ in app/sketch/sketch.ino)
WINDOW_S = 2.0           # one prediction looks at this much data...
HOP_S = 1.0              # ...and a new one is made this often
WINDOW = int(WINDOW_S * FS)
HOP = int(HOP_S * FS)

# CSV columns: t in seconds, accel in g, gyro in deg/s
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
COLUMNS = ["t"] + CHANNELS

# Model 1 looks at the dynamic peak of one accelerometer axis. "auto" lets
# tools/train.py pick the best axis by cross-validation for your placement.
PEAK_AXIS = "auto"

UDP_PORT = 5005          # board -> laptop sample stream (tools/record.py)
