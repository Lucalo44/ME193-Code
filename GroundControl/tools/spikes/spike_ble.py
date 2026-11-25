#!/usr/bin/env python3
"""Phase 0 spike: can the UNO Q's Linux host see and drive a LEGO Double Motor?

Runs on the board HOST (not in the app container, which has no D-Bus socket),
in the venv made for the LEGO service:

    /home/arduino/gc-lego-venv/bin/python spike_ble.py            # scan 8 s
    /home/arduino/gc-lego-venv/bin/python spike_ble.py --connect  # + 20 timed RPCs

--connect uses the purple 0998 Connection Card from the sumo robot by default.
"""
import argparse
import asyncio
import statistics
import time

from bleak import BleakScanner

import legoeducation as le
from legoeducation.basic_ble import LEGO_COMPANY_ID, SERVICE_UUID

P = argparse.ArgumentParser()
P.add_argument("--connect", action="store_true")
P.add_argument("--card", default="0998")
P.add_argument("--seconds", type=float, default=8)
A = P.parse_args()

KINDS = {le.PRODUCT_GROUP_DEVICE_SINGLE_MOTOR: "Single Motor",
         le.PRODUCT_GROUP_DEVICE_DOUBLE_MOTOR: "Double Motor",
         le.PRODUCT_GROUP_DEVICE_COLOR_SENSOR: "Color Sensor",
         le.PRODUCT_GROUP_DEVICE_CONTROLLER: "Controller"}


async def scan():
    found = {}

    def on_adv(dev, adv):
        data = adv.manufacturer_data.get(LEGO_COMPANY_ID)
        if data and len(data) >= 5:
            product = (data[0] << 8) | data[1]
            found[dev.address] = (KINDS.get(product, product), dev.name, adv.rssi,
                                  f"{data[3] | (data[4] << 8):04d}")

    async with BleakScanner(detection_callback=on_adv, service_uuids=[SERVICE_UUID]):
        await asyncio.sleep(A.seconds)
    return found


t0 = time.time()
try:
    found = asyncio.run(scan())
    print(f"PASS  BLE scan from host ({time.time() - t0:.1f}s): {len(found)} LEGO device(s)")
    for addr, (kind, name, rssi, serial) in found.items():
        print(f"      {kind:<13} {name!s:<20} {addr}  rssi {rssi}  card {serial}")
except Exception as e:  # noqa: BLE001
    print(f"FAIL  BLE scan from host: {e!r}")
    raise SystemExit(1)

if A.connect:
    m = le.DoubleMotor()
    t0 = time.time()
    m.connect(card_serial=A.card, card_color=le.LEGO_COLOR_PURPLE, device_notification_delay=20)
    print(f"{'PASS' if m.connected else 'FAIL'}  connect Double Motor card {A.card} "
          f"({time.time() - t0:.1f}s)")
    if m.connected:
        rtts = []
        for _ in range(20):
            t = time.perf_counter()
            m.motor_set_speed(30, motor=le.MOTOR_BOTH, blocking=True)
            rtts.append((time.perf_counter() - t) * 1000)
        print(f"PASS  20 blocking RPCs: median {statistics.median(rtts):.0f} ms, "
              f"min {min(rtts):.0f}, max {max(rtts):.0f}  -> send-ahead ~{statistics.median(rtts) / 2:.0f} ms")
        st = m.motor
        print(f"      motor state A abs={st[0].absolutePosition} B abs={st[1].absolutePosition}")
        m.motor_run_to_absolute_position(0, motor=le.MOTOR_BOTH, speed=30)
        m.beep(frequency=880)
        time.sleep(0.2)
        m.disconnect()
