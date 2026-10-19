#!/usr/bin/env python3
"""Offline checks for the sensor fusion and decision tree (no hardware).
    python test_fusion.py
"""
import legoeducation as le

import sumo
from sumo import Inputs, RingMonitor, decide

BLUE, FLOOR = le.LEGO_COLOR_BLUE, le.LEGO_COLOR_WHITE
DT = 0.02


def run(ring, readings, contact=False, t0=0.0):
    t = t0
    for color, refl in readings:
        ring.update(color, refl, contact, t)
        t += DT
    return t


def inputs(**kw):
    base = dict(match_on=True, out=False, on_tape=False, switch=False, button_tap=False,
                flipper_ready=True, lever_left=0, lever_right=0)
    return Inputs(**{**base, **kw})


def test_flicker_ignored():
    r = RingMonitor()
    run(r, [(FLOOR, 50)] * 10 + [(BLUE, 20)] + [(FLOOR, 50)] * 10)
    assert not r.on_tape and not r.out


def test_enters_and_leaves_with_hysteresis():
    r = RingMonitor()
    run(r, [(BLUE, 20)] * 4)
    assert r.on_tape
    run(r, [(FLOOR, 50)] * 6)
    assert not r.on_tape


def test_out_after_timeout():
    r = RingMonitor()
    run(r, [(BLUE, 20)] * int(sumo.OUT_TAPE_S / DT + 5))
    assert r.out


def test_contact_shortens_timeout():
    n = int((sumo.OUT_TAPE_PUSHED_S + 0.1) / DT)
    free, pushed = RingMonitor(), RingMonitor()
    run(free, [(BLUE, 20)] * n)
    run(pushed, [(BLUE, 20)] * n, contact=True)
    assert not free.out and pushed.out


def test_reflection_gates_color(monkeypatch=None):
    old = sumo.TAPE_REFLECTION
    sumo.TAPE_REFLECTION = (10, 30)
    try:
        both, color_only, refl_only = RingMonitor(), RingMonitor(), RingMonitor()
        run(both, [(BLUE, 20)] * 6)
        run(color_only, [(BLUE, 90)] * 12)   # color alone still gets there, slower
        run(refl_only, [(FLOOR, 20)] * 12)   # reflection alone never does
        assert both.on_tape and color_only.on_tape and not refl_only.on_tape
    finally:
        sumo.TAPE_REFLECTION = old


def test_tree():
    assert decide(inputs(on_tape=True)).state == "ESCAPE"
    d = decide(inputs(on_tape=True, switch=True))
    assert d.state == "ESCAPE_FLIP" and d.fire and d.left == d.right == sumo.ESCAPE_DRIVE
    assert decide(inputs(on_tape=True, switch=True, flipper_ready=False)).state == "ESCAPE"
    assert decide(inputs(out=True, on_tape=True)).state == "LOST"
    assert decide(inputs(switch=True)).state == "AUTO_FLIP"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
