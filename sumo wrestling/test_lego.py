#!/usr/bin/env python3
"""
test_lego.py -- connect to ONE LEGO device and check it works, before
running sumo.py with all three.

    python test_lego.py motor        # wiggles the flipper a little, then beeps
    python test_lego.py sensor       # prints the color it sees (Ctrl+C to stop)
    python test_lego.py controller   # prints lever positions + button (Ctrl+C)

Add --card <serial> to connect to the device with that Connection Card on it
(the number printed on the card). Without --card it connects to the first
device of that type it finds -- fine at home, risky in a room full of teams.
"""

import argparse
import time

import legoeducation as le

from sumo import FLIP_MIRRORED, NOTIFY_MS, connect

DEVICES = {
    "motor": (le.DoubleMotor, "Double Motor (flipper)"),
    "sensor": (le.ColorSensor, "Color Sensor"),
    "controller": (le.Controller, "Controller"),
}


def test_motor(motor):
    print("Wiggling the flipper 30 degrees up and back at 30% speed...")
    if FLIP_MIRRORED:
        motor.movement_move_for_degrees(30, direction=le.MOVEMENT_MOVE_DIRECTION_FORWARD, speed=30)
        motor.movement_move_for_degrees(30, direction=le.MOVEMENT_MOVE_DIRECTION_BACKWARD, speed=30)
    else:
        motor.motor_run_for_degrees(30, motor=le.MOTOR_BOTH, speed=30)
        motor.motor_run_for_degrees(30, motor=le.MOTOR_BOTH, speed=30,
                                    direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    motor.beep(frequency=660)
    print("Done. Did the paddle lift UP first? If it went down, swap FORWARD/BACKWARD "
          "in Flipper._move in sumo.py. If the two sides fought each other, "
          "flip FLIP_MIRRORED.")


def test_sensor(sensor):
    print("Move the sensor over the floor and the blue tape. Ctrl+C to stop.")
    while True:
        s = sensor.sensor
        name = str(le.LEGO_COLOR_NAME_MAP.get(s.color, s.color)).removeprefix("LEGO_COLOR_")
        print(f"  color {name!s:<10} reflection {s.reflection}")
        time.sleep(0.25)


def test_controller(pad):
    print("Move the levers and press the button. Ctrl+C to stop.")
    while True:
        s = pad.sensor
        button = "PRESSED" if pad.button.state == le.BUTTON_STATE_PRESSED else "-"
        print(f"  left {s.leftPercent:>4}  right {s.rightPercent:>4}  button {button}")
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser(description="Test one LEGO device")
    parser.add_argument("device", choices=DEVICES)
    parser.add_argument("--card", help="Connection Card serial printed on the card")
    args = parser.parse_args()

    cls, name = DEVICES[args.device]
    device = connect(cls(), name, args.card)
    try:
        {"motor": test_motor, "sensor": test_sensor, "controller": test_controller}[args.device](device)
    except KeyboardInterrupt:
        pass
    finally:
        device.disconnect()


if __name__ == "__main__":
    main()
