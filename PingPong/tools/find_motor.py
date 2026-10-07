#!/usr/bin/env python3
"""
tools/find_motor.py -- list every LEGO Education device advertising nearby, with
its Connection Card color and serial, and whether it matches CARD_COLOR /
CARD_SERIAL in config.py.

    python tools/find_motor.py              # 8 s scan
    python tools/find_motor.py --seconds 15

Turn the motor on (and make sure no other program is connected to it -- a
connected motor stops advertising). Copy the color and serial printed for your
motor into config.py.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from bleak import BleakScanner  # noqa: E402
from legoeducation.basic_ble import SERVICE_UUID_LOWER, BasicBLE  # noqa: E402

import config as C  # noqa: E402
from hardware.paddle import card_filter, color_name, describe_card_filter  # noqa: E402

PRODUCTS = {513: "Double Motor"}   # legoeducation PRODUCT_GROUP_DEVICE_DOUBLE_MOTOR


async def scan(seconds: float) -> dict:
    found = {}

    def on_adv(device, adv):
        if SERVICE_UUID_LOWER not in [u.lower() for u in (adv.service_uuids or [])]:
            return
        product, color, serial = BasicBLE._extract_manufacturer_info(adv)
        found[device.address] = (device.name or "?", product, color, serial, adv.rssi)

    async with BleakScanner(on_adv):
        await asyncio.sleep(seconds)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    try:
        want_color, want_serial = card_filter()
        print(f"config.py asks for {describe_card_filter()}.")
    except ValueError as exc:
        print(f"config.py problem: {exc}")
        want_color = want_serial = None
    print(f"Scanning for {args.seconds:.0f} s...\n")
    found = asyncio.run(scan(args.seconds))
    if not found:
        print("No LEGO Education devices found. Is the motor on, nearby, and not connected to another program?\n"
              "(macOS: allow Bluetooth for your terminal in System Settings > Privacy & Security > Bluetooth.)")
        return 1
    print(f"{'device':22s} {'card color':10s} {'serial':6s} {'signal':>6s}  matches config?")
    for addr, (name, product, color, serial, rssi) in sorted(found.items(), key=lambda kv: -(kv[1][4] or -999)):
        kind = PRODUCTS.get(product, f"product {product}")
        serial_txt = f"{serial:04d}" if serial is not None else "?"
        ok = (product == 513
              and (want_color is None or color == want_color)
              and (want_serial is None or serial_txt == want_serial))
        why = "YES" if ok else ("no (not a Double Motor)" if product != 513 else "no")
        print(f"{(name + ' / ' + kind)[:22]:22s} {color_name(color):10s} {serial_txt:6s} {rssi:>6} dBm  {why}")
    motors = [v for v in found.values() if v[1] == 513]
    if motors and not any((want_color is None or m[2] == want_color) and
                          (want_serial is None or f"{m[3]:04d}" == want_serial) for m in motors):
        name, product, color, serial, _ = max(motors, key=lambda m: m[4] or -999)
        print(f"\nNo motor matches. For the strongest-signal motor, set in config.py:\n"
              f'    CARD_COLOR = "{color_name(color)}"\n    CARD_SERIAL = "{serial:04d}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
