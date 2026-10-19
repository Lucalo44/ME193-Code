#!/usr/bin/env python3
"""
test_lego.py -- connect to ONE LEGO device and check it works, before
running sumo.py with all three.

    python test_lego.py motor        # wiggles the flipper a little, then beeps
    python test_lego.py sensor       # prints the color it sees (Ctrl+C to stop)
    python test_lego.py imu          # prints the Double Motor IMU: tap/shove it to tune IMU_HIT_JERK
    python test_lego.py controller   # prints lever positions + button (Ctrl+C)
    python test_lego.py drive        # levers drive the UnoQ wheels over MQTT (Ctrl+C)
    python test_lego.py scan         # list every LEGO device nearby + its card

It connects to the device that has our purple 0998 Connection Card tapped
onto it (CARD_COLOR / CARD_SERIAL in sumo.py). For a different card:

    python test_lego.py motor --card 1234 --card-color red
"""

import argparse
import asyncio
import time

from bleak import BleakScanner

import legoeducation as le

from sumo import (BROKER, CARD_COLORS, FLIP_MIRRORED, PORT, TOPIC_BASE, SwitchLink,
                  add_card_args, connect, levers_to_drive)

DEVICES = {
    "motor": (le.DoubleMotor, "Double Motor (flipper)"),
    "imu": (le.DoubleMotor, "Double Motor (IMU)"),
    "sensor": (le.ColorSensor, "Color Sensor"),
    "controller": (le.Controller, "Controller"),
    "drive": (le.Controller, "Controller"),
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


def test_imu(motor):
    """Shows the jerk the way sumo.py computes it. Tap the robot, shove it, run
    the wheels, and pick IMU_HIT_JERK between 'quiet' and 'real hit'."""
    from sumo import ImuMonitor
    imu = ImuMonitor()
    print("Tap and shove the robot. Peak jerk is since the last line. Ctrl+C to stop.")
    peak = 0.0
    last = 0.0
    while True:
        now = time.monotonic()
        imu.update(motor, now)
        peak = max(peak, imu.jerk)
        if now - last >= 0.25:
            s = motor.imu_device
            print(f"  peak jerk {peak:8.0f}  {'HIT' if imu.hit(now) else '   '}"
                  f"  pitch {s.pitch}  roll {s.roll}  gesture {motor.imu_gesture.gesture}")
            peak, last = 0.0, now
        time.sleep(0.02)


def test_controller(pad):
    print("Move the levers and press the button. Ctrl+C to stop.")
    while True:
        s = pad.sensor
        button = "PRESSED" if pad.button.state == le.BUTTON_STATE_PRESSED else "-"
        print(f"  left {s.leftPercent:>4}  right {s.rightPercent:>4}  button {button}")
        time.sleep(0.25)


def test_drive(pad):
    """Controller levers -> UnoQ wheels, exactly as sumo.py's TELEOP does,
    without the color sensor or flipper."""
    link = SwitchLink(BROKER, PORT, TOPIC_BASE)
    print("Wheels off the table! Push the levers to drive. Ctrl+C to stop.")
    try:
        last_print = 0.0
        while True:
            s = pad.sensor
            levers = [0 if v != v else int(v) for v in (s.leftPercent, s.rightPercent)]  # NaN -> 0
            left, right = levers_to_drive(*levers)
            link.drive(left, right)
            if time.monotonic() - last_print >= 0.25:
                print(f"  levers {levers[0]:>4} {levers[1]:>4}  ->  wheels {left:>4} {right:>4}")
                last_print = time.monotonic()
            time.sleep(0.05)
    finally:
        link.close()  # sends a final stop


def scan(seconds=8):
    """List every LEGO device advertising nearby, with the card it reports."""
    from legoeducation.basic_ble import LEGO_COMPANY_ID, SERVICE_UUID
    from legoeducation.color_map import _firmware_to_app

    kinds = {le.PRODUCT_GROUP_DEVICE_SINGLE_MOTOR: "Single Motor",
             le.PRODUCT_GROUP_DEVICE_DOUBLE_MOTOR: "Double Motor",
             le.PRODUCT_GROUP_DEVICE_COLOR_SENSOR: "Color Sensor",
             le.PRODUCT_GROUP_DEVICE_CONTROLLER: "Controller"}
    color_names = {c: n for n, c in CARD_COLORS.items()}
    found = {}

    def on_adv(device, adv):
        data = adv.manufacturer_data.get(LEGO_COMPANY_ID)
        if data is None or len(data) < 5:
            return
        product = (data[0] << 8) | data[1]
        color = _firmware_to_app(data[2])
        found[device.address] = (kinds.get(product, f"product {product}"), device.name,
                                 color_names.get(color, f"none ({color})"),
                                 f"{data[3] | (data[4] << 8):04d}")

    async def run():
        async with BleakScanner(detection_callback=on_adv, service_uuids=[SERVICE_UUID]):
            await asyncio.sleep(seconds)

    print(f"Scanning {seconds} s for LEGO devices (devices already connected "
          f"to a program or the LEGO app don't show up)...")
    asyncio.run(run())
    if not found:
        print("  nothing found -- are they turned on? Is Bluetooth allowed for this terminal?")
    for kind, name, color, serial in sorted(found.values()):
        print(f"  {kind:<13} {str(name):<22} card {color:<10} {serial}")


def main():
    parser = argparse.ArgumentParser(description="Test one LEGO device")
    parser.add_argument("device", choices=[*DEVICES, "scan"])
    add_card_args(parser)
    args = parser.parse_args()
    if args.device == "scan":
        return scan()

    cls, name = DEVICES[args.device]
    device = connect(cls(), name, args.card, CARD_COLORS[args.card_color])
    try:
        {"motor": test_motor, "sensor": test_sensor, "imu": test_imu, "controller": test_controller,
         "drive": test_drive}[args.device](device)
    except KeyboardInterrupt:
        pass
    finally:
        device.disconnect()


if __name__ == "__main__":
    main()
