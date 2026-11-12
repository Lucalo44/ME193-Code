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
    """Mirrors every call to the board, and keeps the virtual state so the UI still works.

    All Bridge traffic runs on one background thread, so the camera loop never waits on the
    microcontroller. The Linux <-> STM32 link can corrupt a message (a flipped bit once turned
    "button_pressed" into "button_`ressed"), and right after a flash the STM32 needs a few
    seconds before it answers. So the thread:
      * stays quiet for STARTUP_QUIET_S, then probes until the board answers;
      * polls the button at 10 Hz (the sketch latches presses, so none are missed);
      * keeps the bolt's *desired* state and resends it until the board confirms it;
      * sends only the newest LED matrix frame;
      * backs off to one probe per second while the link is down.
    """

    BUTTON_POLL_S = 0.1
    RETRY_S = 1.0
    STARTUP_QUIET_S = 5.0

    def __init__(self):
        super().__init__()
        import threading
        from arduino.app_utils import Bridge, Frame   # only exist on the board
        self.bridge, self.Frame = Bridge, Frame
        self._lock = threading.Lock()
        self._hw_pressed = False
        self._bolt_want, self._bolt_sent = None, None
        self._frame_want, self._frame_sent = None, None
        self._t_fail, self._warned = None, False
        self.errors, self._first_err = 0, ""
        self.link_up = False
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- called from the safe loop: never blocks ----
    def button_pressed(self):
        with self._lock:
            hw, self._hw_pressed = self._hw_pressed, False
        return super().button_pressed() or hw

    def unlock(self):
        super().unlock()
        with self._lock:
            self._bolt_want, self._bolt_sent = 0, None

    def lock(self):
        super().lock()
        with self._lock:
            self._bolt_want, self._bolt_sent = 1, None

    def tick(self, now=None, hold=None):
        f = matrix_frame(self.leds, self.digits, hold, time.time() if now is None else now)
        with self._lock:
            self._frame_want = f

    # ---- the Bridge thread ----
    def _call(self, name, *args, timeout=1.5):
        try:
            out = self.bridge.call(name, *args, timeout=timeout)
        except Exception as e:                      # RpcError, TimeoutError, connection loss
            now = time.time()
            self.link_up = False
            if self._t_fail is None:
                self._t_fail, self._first_err = now, f"{name}: {e}"
                self.errors += 1
            elif not self._warned and now - self._t_fail > 15:
                self._warned = True
                print("[bridge] the microcontroller has not answered for 15 s. If its red LED is "
                      "blinking, power-cycle the board (unplug ~5 s).")
            raise
        if self._t_fail is not None:
            down = time.time() - self._t_fail
            if down > 1.0:                          # one-off bit errors are only counted
                print(f"[bridge] link back after {down:.1f} s (first error: {self._first_err})")
            self._t_fail, self._warned = None, False
        self.link_up = True
        return out

    def _worker(self):
        time.sleep(self.STARTUP_QUIET_S)            # let a freshly flashed STM32 finish booting
        print("[bridge] connecting to the microcontroller")
        while True:
            t0 = time.time()
            try:
                if int(self._call("button_pressed") or 0) == 1:
                    with self._lock:
                        self._hw_pressed = True
                with self._lock:
                    bolt = self._bolt_want if self._bolt_sent != self._bolt_want else None
                    frame = self._frame_want
                    send_frame = frame is not None and (self._frame_sent is None or not np.array_equal(frame, self._frame_sent))
                if bolt is not None:
                    self._call("set_bolt", bolt, timeout=3)
                    with self._lock:
                        if self._bolt_want == bolt:
                            self._bolt_sent = bolt
                if send_frame:
                    self._call("draw", self.Frame(frame).to_board_bytes())
                    self._frame_sent = frame
                time.sleep(max(0.0, self.BUTTON_POLL_S - (time.time() - t0)))
            except Exception:
                self._frame_sent = None             # resend the matrix once the link is back
                time.sleep(self.RETRY_S)            # link down: one probe per second
