"""The safe's hardware: one button, an LED ring, and a bolt.

VirtualSafe keeps it all on screen (the button is clicked in the UI or pressed with SPACE).
UnoQSafe adds the same calls over the UNO Q's Bridge to the STM32 sketch. The sketch side is
not written yet; it needs to `Bridge.provide()` these three functions:
    button_pressed() -> "1" once per press, else "0"
    set_leds(mode, n)   mode in off / armed / progress / granted / denied / lockout
    set_bolt(locked)    "1" = locked, "0" = open (servo or solenoid)
"""


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


class UnoQSafe(VirtualSafe):
    """Mirrors every call to the board, and keeps the virtual state so the UI still works."""

    def __init__(self):
        super().__init__()
        from arduino.app_utils import Bridge   # only exists on the board
        self.bridge = Bridge

    def button_pressed(self):
        return super().button_pressed() or str(self.bridge.call("button_pressed")) == "1"

    def set_leds(self, mode, digits=0):
        super().set_leds(mode, digits)
        self.bridge.call("set_leds", mode, str(digits))

    def unlock(self):
        super().unlock()
        self.bridge.call("set_bolt", "0")

    def lock(self):
        super().lock()
        self.bridge.call("set_bolt", "1")
