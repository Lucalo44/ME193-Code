"""The safe's hardware: a start button, status LEDs, and a servo bolt.

VirtualSafe keeps it all on screen (the button is clicked in the UI or pressed with SPACE).
UnoQSafe drives the real parts through the UNO Q's Bridge to ArduinoUNOq/safe/sketch:
    button_pressed() -> 1 once per press (latched on the MCU, so short taps aren't missed)
    set_bolt(locked)    1 = locked, 0 = open (servo)
    draw(frame)         8 x 13 LED matrix, brightness 0-7 per pixel
The LED matrix stands in for the "LEDs light up" step; the screens are drawn here.
"""
import time

import numpy as np

ROWS, COLS = 8, 13


class VirtualSafe:
    def __init__(self):
        self.locked = True
        self.leds = "off"
        self.digits = 0
        self._pressed = False

    def press(self):                    # called by the on-screen button
        self._pressed = True

    def button_pressed(self):
        pressed, self._pressed = self._pressed, False
        return pressed

    def set_leds(self, mode, digits=0):
        self.leds, self.digits = mode, digits

    def unlock(self):
        self.locked = False

    def lock(self):
        self.locked = True

    def tick(self, now, hold=None):
        """Called every loop with run_safe's hold info; hardware uses it for the LED matrix."""


# 5 x 7 digits for the confirm flash ('#' = lit)
DIGITS = {
    1: ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    2: [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    3: [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."],
    4: ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    5: ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
}


def matrix_frame(mode, digits, hold, t):
    """The LED matrix screen for each safe state, as an 8 x 13 array of 0-7.
    hold: run_safe's hold info (progress, and "flash" = a step that was just confirmed)."""
    f = np.zeros((ROWS, COLS), np.uint8)
    hold = hold or {}
    flash = hold.get("flash")
    if mode in ("armed", "progress") and flash is not None and int(flash) in DIGITS:
        for r, row in enumerate(DIGITS[int(flash)]):  # the confirmed number, big, centered
            for c, ch in enumerate(row):
                if ch == "#":
                    f[r, 4 + c] = 7
        if hold.get("flash_text") == "HEARD":         # spoken step: sound bars on both sides
            f[2:5, 1] = f[1:6, 2] = f[2:5, 10] = f[1:6, 11] = 3
        return f
    hold_progress = hold.get("progress", 0.0)
    if mode in ("armed", "progress"):
        for i, c0 in enumerate((1, 5, 9)):              # three boxes = three steps
            if i < digits:
                f[0:3, c0:c0 + 3] = 7
            else:
                f[0:3, c0:c0 + 3] = 2
                f[1, c0 + 1] = 0
        f[6, :] = 1                                     # hold bar: fills over the 3 s hold
        f[6, :int(round(COLS * min(1.0, hold_progress)))] = 7
    elif mode == "granted":                             # check mark
        for r, c in [(4, 2), (5, 3), (6, 4), (5, 5), (4, 6), (3, 7), (2, 8), (1, 9), (0, 10)]:
            f[r, c] = 7
    elif mode in ("denied", "lockout"):                 # X: blinks on denied, glows on lockout
        level = (7 if int(t * 6) % 2 else 0) if mode == "denied" else int(2 + 1.5 * np.sin(t * 2))
        for i in range(ROWS):
            f[i, 2 + i] = f[i, 10 - i] = level
    elif mode == "open":                                # open padlock: shackle swung up and off
        f[3:8, 4:9] = 3
        f[0, 6:10] = 3
        f[1:3, 6] = f[1:3, 9] = 3
        f[5, 6] = 0
    else:                                               # locked + idle: closed padlock, dim
        f[3:8, 4:9] = 2
        f[0, 5:8] = 2
        f[1:3, 4] = f[1:3, 8] = 2
        f[5, 6] = 0
    return f


class UnoQSafe(VirtualSafe):
    """Mirrors every call to the board, and keeps the virtual state so the UI still works."""

    def __init__(self):
        super().__init__()
        from arduino.app_utils import Bridge, Frame   # only exist on the board
        self.bridge, self.Frame = Bridge, Frame
        self._last = None

    def button_pressed(self):
        return super().button_pressed() or int(self.bridge.call("button_pressed") or 0) == 1

    def unlock(self):
        super().unlock()
        self.bridge.call("set_bolt", 0)

    def lock(self):
        super().lock()
        self.bridge.call("set_bolt", 1)

    def tick(self, now=None, hold=None):
        f = matrix_frame(self.leds, self.digits, hold, time.time() if now is None else now)
        if self._last is None or not np.array_equal(f, self._last):   # only send changes
            self.bridge.call("draw", self.Frame(f).to_board_bytes())
            self._last = f
