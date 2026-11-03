# Safe

The goal here is to build a properly fun lock around the Arduino UNO Q. You walk up to the safe and push a button, which lights the LEDs. Then you hold three finger counts (each 1–5) under a USB camera, and the safe either opens or stays shut. The physical safe is still in progress. Until it exists, the safe is a virtual window on my laptop, drawn by [ui.py](ui.py).

## How it works

In order to keep the whole stack to PyTorch and OpenCV (both of which install cleanly on the UNO Q's Debian side), I skipped hand-landmark libraries. Instead I crop a fixed square "place your hand here" box out of each frame and classify it directly. All of the crop and input logic lives in [preprocess.py](preprocess.py). The collector, the trainer and the safe all import it, so the nets see identical pixels at training time and at the lock.

There are two small CNNs, both defined in [models.py](models.py):

1. The gesture net takes a 96px RGB crop and outputs `none / 1 / 2 / 3 / 4 / 5`. The `none` class matters more than it looks. It is how the safe knows your hand has left the box between digits, which in turn is what lets a code like 3-3-1 work.
2. The owner net decides `me / not_me` from the palm's crease lines. A raw webcam image is mostly lighting and skin tone, which says nothing useful about whose hand it is. To get around that, the crop is converted to gray and equalized with CLAHE. A morphological black-hat then keeps only the dark features narrower than a 9px kernel, and that is roughly what a crease is. The net sees the equalized gray and this crease map as two channels. Of note, the owner data is never mirrored during augmentation, since a flipped palm is effectively someone else's palm.

[run_safe.py](run_safe.py) turns the noisy per-frame predictions into digits. A digit only counts after 12 confident frames in a row (~0.4s), and the box then has to read `none` for 6 frames before the next digit is accepted. With owner check on, P(me) is averaged over each digit's frames, and all three digits have to clear 0.85. Three wrong attempts in a row lock the keypad for 30s. Every one of these numbers lives in [config.py](config.py).

## Running it

```bash
pip install -r requirements.txt
```

1. Set the code with [set_password.py](set_password.py), e.g. `python3 set_password.py 4 2 5`. Only a salted hash is stored, in a git-ignored `password.json`.
2. Gesture-only safe:
   1. Record data with `python3 collect_data.py --task gesture`. Keys 0–5 pick the label (0 = none), and SPACE starts or stops a session. Move and tilt your hand while recording. For `none`, record an empty box, a hand half in or out, and a fist.
   2. Train with `python3 train.py --task gesture`. Train/val is split by recording session, so the val accuracy isn't just neighboring frames of the training set.
   3. Run `python3 run_safe.py`. Click PRESS or hit SPACE, then show your three numbers.
3. Owner-only safe (palm creases):
   1. Record with `python3 collect_data.py --task owner`. Use `m` for my hand and `n` for anyone else's. Keep the palm facing the camera while cycling through 1–5. The more different `not_me` hands, the stricter the check gets.
   2. Train with `python3 train.py --task owner`. Read the printed false-accept rate, since that is the number that matters for a lock.
   3. Run `python3 run_safe.py --owner-check`, or press `o` in the window to toggle the check on and off live.
4. On the UNO Q, `python3 run_safe.py --hardware unoq --headless` sends the button, LEDs and bolt through the Bridge via [safe_io.py](safe_io.py). The sketch side (`button_pressed`, `set_leds`, `set_bolt`) is not written yet, and neither is the actual safe.

`data/` is git-ignored, since it is a pile of pictures of my hand.
