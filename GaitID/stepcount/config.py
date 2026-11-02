"""Step counter constants. Sensor layout, sample rate, network and training defaults."""

from gaitid import config as G

FS = 50                     # everything is resampled onto this grid (Hz)
UDP_PORT = G.UDP_PORT       # the UNO Q's MPU-6050 stream (app/python/main.py, STREAM_TO set)

# ---- sensors -----------------------------------------------------------------
# "mpu":  MPU-6050 on the UNO Q, worn on the RIGHT ankle. g and deg/s, ~50 Hz over UDP.
# "lego": LEGO Education Double Motor IMU, worn on the LEFT ankle. Raw counts, ~66 Hz over BLE.
# One IMU per ankle means every foot strike lands right under a sensor.
SENSORS = ["mpu", "lego"]
SENSOR_CHANNELS = {
    "mpu": ["ax", "ay", "az", "gx", "gy", "gz"],
    "lego": ["ax", "ay", "az", "gx", "gy", "gz", "pitch", "roll"],   # yaw is heading: arbitrary zero, drifts
}
# Per sensor the network also gets |a| and |gyro|, plus a 1/0 "sensor present" channel.
DERIVED = ["amag", "gmag", "present"]

# Double Motor raw units (same values as PingPong/config.py)
LEGO_CARD_COLOR = None      # e.g. "purple"; None = first Double Motor found. Check with PingPong/tools/find_motor.py
LEGO_CARD_SERIAL = None     # e.g. "0998"
LEGO_NOTIFY_MS = 15         # fastest the motor allows (~66 Hz)
LEGO_ACCEL_RAW_PER_G = 1000.0
LEGO_GYRO_RAW_PER_DPS = 1.0
LEGO_ANGLE_RAW_PER_DEG = 10.0

# ---- labels --------------------------------------------------------------------
STEP_SIGMA_S = 0.05         # each labelled step becomes a Gaussian bump this wide (area 1)
MIN_STEP_GAP_S = 0.22       # no human steps faster than ~4.5 per second
MIN_STRIDE_S = 0.5          # one foot's swing-to-swing time can't be shorter (240 steps/min)

# ---- network & training ---------------------------------------------------------
WINDOW_S = 8.0              # training crop length
HIDDEN = 48                 # conv channels
DILATIONS = [1, 2, 4, 8, 16]
KERNEL = 5                  # receptive field = 1 + (KERNEL-1) * sum(DILATIONS) samples (~2.5 s)
COUNT_WEIGHT = 0.05         # weight of the per-window step-count error in the loss
LIVE_MARGIN_S = 1.5         # live counting commits density this far behind the newest sample
