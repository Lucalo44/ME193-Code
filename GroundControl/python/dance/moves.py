"""Move library. Every move is defined in BEATS, never seconds:

    move(start_beat, beats, intensity, mirror) -> [Keyframe, ...]

Keyframe actions
  car   : wheels {l, r}  signed wheel PWM -255..255
          matrix {icon}  or {digit} or {clear: True}
  lego  : pose   {a, b, speed}   absolute arm angles (deg) for the "arms" build
          run    {a, b}          continuous speed % per output (windmill)
          tank   {l, r}          speed % per side for the "wheels" build
          stop   {}

Physical limits live here too, so the compiler can't emit anything the hardware
can't do (wheel PWM is clamped, LEGO poses stay inside ARM_RANGE).
"""
from dataclasses import dataclass, field

CAR_MOVES = ["shimmy", "spin", "step", "wiggle", "freeze", "bow"]
LEGO_MOVES = ["wave", "clap", "pump", "point", "windmill", "freeze", "bow"]
MODES = ["unison", "mirror", "call_response", "canon", "solo_car", "solo_lego"]

WHEEL_MAX = 255
ARM_RANGE = (-30, 130)        # deg; 0 = arm down, positive = raised
LEGO_SPEED_MAX = 100


@dataclass
class Keyframe:
    t: float                  # beat (relative to the first downbeat)
    action: str
    params: dict = field(default_factory=dict)
    move: str = ""

    def as_dict(self):
        return {"t": round(self.t, 4), "action": self.action, "params": self.params, "move": self.move}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _w(s):
    return int(_clamp(s, -WHEEL_MAX, WHEEL_MAX))


def _arm(a):
    return int(_clamp(a, *ARM_RANGE))


# ------------------------------------------------------------------ car
def car_move(name, start, beats, intensity=0.7, mirror=False):
    s = 130 + 120 * intensity
    sg = -1 if mirror else 1
    k = []

    def wheels(t, l, r):
        k.append(Keyframe(start + t, "wheels", {"l": _w(l), "r": _w(r)}, name))

    if name == "shimmy":
        for b in range(beats):
            wheels(b, sg * s, -sg * s)
            wheels(b + 0.25, 0, 0)
            wheels(b + 0.5, -sg * s, sg * s)
            wheels(b + 0.75, 0, 0)
    elif name == "spin":
        wheels(0, sg * s, -sg * s)
    elif name == "step":
        for b in range(0, beats, 2):
            wheels(b, s, s)
            wheels(b + 0.5, 0, 0)
            if b + 1 < beats:
                wheels(b + 1, -s, -s)
                wheels(b + 1.5, 0, 0)
    elif name == "wiggle":
        w = 0.7 * s
        for q in range(beats * 4):
            d = sg if q % 2 == 0 else -sg
            wheels(q * 0.25, d * w, -d * w)
    elif name == "freeze":
        wheels(0, 0, 0)
    elif name == "bow":
        wheels(0, -0.6 * s, -0.6 * s)
        wheels(0.5, 0, 0)
        k.append(Keyframe(start + 0.5, "matrix", {"icon": "heart"}, name))
    else:
        raise ValueError(f"unknown car move {name}")
    return k


# ------------------------------------------------------------------ lego
def lego_move(name, start, beats, intensity=0.7, mirror=False, build="arms"):
    if build == "wheels":
        return _lego_wheels(name, start, beats, intensity, mirror)
    sp = int(50 + 50 * intensity)
    amp = 0.6 + 0.4 * intensity
    k = []

    def pose(t, a, b, speed=sp):
        if mirror:
            a, b = b, a
        k.append(Keyframe(start + t, "pose", {"a": _arm(a), "b": _arm(b), "speed": speed}, name))

    if name == "wave":
        for b in range(beats):
            pose(b, 90 * amp, 0)
            pose(b + 0.5, 20, 0)
    elif name == "clap":
        for b in range(beats):
            pose(b, 60 * amp, 60 * amp)
            pose(b + 0.5, 0, 0)
    elif name == "pump":
        for b in range(beats):
            pose(b, 110 * amp, 110 * amp, min(LEGO_SPEED_MAX, sp + 20))
            pose(b + 0.5, 10, 10, min(LEGO_SPEED_MAX, sp + 20))
    elif name == "point":
        pose(0, 100, 0)
    elif name == "windmill":
        v = int(40 + 50 * intensity)
        a, b = (v, -v) if not mirror else (-v, v)
        k.append(Keyframe(start, "run", {"a": a, "b": b}, name))
    elif name == "freeze":
        k.append(Keyframe(start, "stop", {}, name))
    elif name == "bow":
        pose(0, 120, 120)
        pose(min(1.5, beats - 0.5), 0, 0)
    else:
        raise ValueError(f"unknown lego move {name}")
    return k


def _lego_wheels(name, start, beats, intensity, mirror):
    """Same move names for a LEGO bot built on two wheels (outputs = left/right)."""
    v = int(40 + 50 * intensity)
    sg = -1 if mirror else 1
    k = []

    def tank(t, l, r):
        k.append(Keyframe(start + t, "tank", {"l": int(_clamp(l, -100, 100)),
                                              "r": int(_clamp(r, -100, 100))}, name))

    if name == "wave":
        for b in range(beats):
            tank(b, sg * v, -sg * v)
            tank(b + 0.5, -sg * v, sg * v)
    elif name in ("clap", "pump"):
        for b in range(beats):
            tank(b, v, v)
            tank(b + 0.5, -v, -v)
    elif name == "point":
        tank(0, sg * v, -sg * v)
        tank(0.5, 0, 0)
    elif name == "windmill":
        tank(0, sg * v, -sg * v)
    elif name == "freeze":
        tank(0, 0, 0)
    elif name == "bow":
        tank(0, -v, -v)
        tank(0.5, 0, 0)
    else:
        raise ValueError(f"unknown lego move {name}")
    return k
