"""Stick figures that dance on the 8x13 LED matrix.

One figure (the default) dances both parts: its legs follow the CAR lane and its arms
the LEGO lane. Two figures stand side by side: the left one dances the car's part, the
right one the LEGO's part, and in mirror sections the right one is drawn flipped, so the
partners face each other. dance/styles.py decides how many.

Timing is a 16th-note grid. Each lane has a rhythm (dance/rhythm.py): the pose changes
only on a hit, and the figure flashes to full brightness on the hit and settles to a
dimmer hold a 16th later, so the rhythm reads even when the pose repeats. Beat lights in
the bottom corners mark every beat and stand taller on beat 1.

Poses are written "arms/legs[/dx][/nohead]" (or "bow"); each move has a default cycle of
them, and style profiles bring their own.

    figure_timeline(compiled_score) -> [{"t", "beat", "action": "frame", "params": {"frame": [...104]}}]
"""
from functools import lru_cache

from dance.rhythm import DEFAULT as DEFAULT_RHYTHM
from dance.rhythm import is_hit
from hal import COLS, ROWS

HIT, HOLD, BEAT, DOWNBEAT = 7, 4, 3, 7          # brightness 0-7
SOLO_X, DUO_X = 4, (1, 7)                       # left column of each 5-wide figure
LIGHTS = (0, COLS - 1)                          # beat lights in the bottom corners
DUO_BOUNDS = ((1, 5), (7, COLS - 2))            # each partner keeps to its own half

# Each layer: {row: "5-char string"}; a digit is that LED's brightness (0-7), '.' is off.
# The head is a bright 3x2 block and the body and hands are dimmer, so a raised hand never
# reads as the head (a 1-pixel head did, and the figure looked headless).
HEAD = {0: ".777.", 1: ".777."}
BACK_OF_HEAD = {0: ".444.", 1: ".444."}         # turned away (mid-spin)
TORSO = {2: "..5..", 3: "..5..", 4: "..5.."}
ARMS = {
    "down":     {2: ".555.", 3: "5...5", 4: "5...5"},
    "up":       {0: "3...3", 1: "3...3", 2: ".555."},
    "out":      {2: "55555"},
    "left_up":  {0: "3....", 1: "3....", 2: ".555.", 3: "....5", 4: "....5"},
    "right_up": {0: "....3", 1: "....3", 2: ".555.", 3: "5....", 4: "5...."},
    "clap":     {2: ".555.", 3: ".565."},          # hands together in front of the chest
    "point":    {2: ".5555", 3: "5....", 4: "5...."},
    "side":     {2: "..55.", 3: "...5."},          # seen from the side (turns)
    "hips":     {2: ".555.", 3: "5...5", 4: ".5.5."},
    "frame":    {1: "3...3", 2: "55555"},          # ballroom hold, robot arms
    "robot_dn": {2: "55555", 3: "5...5"},
    "fifth":    {0: "3...3", 2: ".555."},          # ballet: arms rounded overhead
}
LEGS = {
    "stand":  {5: ".5.5.", 6: ".5.5.", 7: ".5.5."},
    "apart":  {5: ".5.5.", 6: "5...5", 7: "5...5"},
    "step_l": {5: ".5.5.", 6: "5..5.", 7: "5..5."},
    "step_r": {5: ".5.5.", 6: ".5..5", 7: ".5..5"},
    "plie":   {5: ".5.5.", 6: "5...5", 7: ".5.5."},  # knees out
    "rise":   {5: "..5..", 6: "..5..", 7: "..5.."},  # releve, legs together
    "kick_r": {5: ".5.55", 6: ".5...", 7: ".5..."},
    "kick_l": {5: "55.5.", 6: "...5.", 7: "...5."},
    "side":   {5: "..5..", 6: "..5..", 7: ".55.."},
}
BOW = {3: "77555", 4: "77..5", 5: "...55", 6: "...55", 7: "...55"}   # bent forward, head down

# Default pose cycles. The car lane leads with the legs, the LEGO lane with the arms.
CAR_FIG = {
    "shimmy": ["out/apart/1", "out/apart/-1"],
    "spin":   ["out/apart", "side/side", "out/apart/0/nohead", "side/side"],
    "step":   ["right_up/step_l", "left_up/step_r"],
    "wiggle": ["up/stand/1", "up/stand/-1"],
    "freeze": ["down/stand"],
    "bow":    ["bow"],
}
LEGO_FIG = {
    "wave":     ["left_up/stand", "down/stand"],
    "clap":     ["clap/plie", "out/stand"],
    "pump":     ["up/apart", "down/plie"],
    "point":    ["point/stand"],
    "windmill": ["left_up/apart", "out/apart", "right_up/apart", "down/apart"],
    "freeze":   ["down/stand"],
    "bow":      ["bow"],
}
DEFAULT_FIG = {"car": CAR_FIG, "lego": LEGO_FIG}
STILL = ("freeze", "bow")


@lru_cache(maxsize=None)
def pose(spec):
    """"frame/plie/-1/nohead" -> {"arms", "legs", "dx", "head", "bow"}; raises on unknown names."""
    if spec == "bow":
        return {"bow": True, "arms": "down", "legs": "stand", "dx": 0, "head": False}
    parts = spec.split("/")
    arms, legs = parts[0], parts[1]
    if arms not in ARMS or legs not in LEGS:
        raise ValueError(f"unknown pose {spec!r}")
    dx = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else 0
    return {"bow": False, "arms": arms, "legs": legs, "dx": dx, "head": "nohead" not in parts}


def render(arms="down", legs="stand", dx=0, bow=False, head=True):
    """One centred figure (kept for callers that draw a single pose)."""
    f = [0] * (ROWS * COLS)
    draw(f, {"arms": arms, "legs": legs, "dx": dx, "bow": bow, "head": head}, SOLO_X, (0, COLS - 1))
    return f


def draw(f, p, x0, bounds, level=7, flip=False, flip_arms=False):
    """Max a 5-wide figure into frame f at column x0, clipped to bounds (lo, hi). Each LED's
    brightness is scaled by level/7, so a held pose dims evenly and the head stays brightest."""
    dx = -p["dx"] if flip else p["dx"]
    if p["bow"]:
        layers = [(BOW, flip)]
    else:
        layers = [(HEAD if p["head"] else BACK_OF_HEAD, False), (TORSO, False),
                  (ARMS[p["arms"]], flip != flip_arms), (LEGS[p["legs"]], flip)]
    lo, hi = bounds
    for layer, rev in layers:
        for r, row in layer.items():
            for i, ch in enumerate(row[::-1] if rev else row):
                c = x0 + i + dx
                if ch != "." and lo <= c <= hi:
                    v = 7 if ch == "#" else int(ch)
                    v = v if level >= 7 else max(1, round(v * level / 7))
                    f[r * COLS + c] = max(f[r * COLS + c], v)
    return f


def _track(blocks, lane, n16, meter):
    """Per 16th: (pose, hit, block) for one lane, or None where the lane has no block."""
    out = [None] * n16
    for bl in blocks:
        if bl["lane"] != lane:
            continue
        cycle = [pose(s) for s in (bl.get("fig") or DEFAULT_FIG[lane][bl["move"]])]
        rhythm = bl.get("rhythm")
        if rhythm is None:
            rhythm = DEFAULT_RHYTHM[lane][bl["move"]]
        if bl["move"] in STILL and not bl.get("fig"):
            rhythm = ""                                    # a plain freeze or bow holds still
        s0, s1 = round(bl["start"] * 4), round((bl["start"] + bl["beats"]) * 4)
        sec0 = round(bl.get("sec_start", bl["start"]) * 4)
        n = -1
        for g in range(s0, min(s1, n16)):
            hit = g == s0 or is_hit(rhythm, g - sec0, meter)
            n += hit
            out[g] = (cycle[n % len(cycle)], hit, bl)
    return out


STAND = pose("down/stand")


def _solo_pose(car, lego):
    """One figure dancing both parts: legs from the car lane, arms from the LEGO lane. When
    one lane is resting (a freeze while the other dances), the dancing lane gets the whole body."""
    cp, _, cb = car or (STAND, False, None)
    lp, _, lb = lego or (STAND, False, None)
    cm, lm = (cb or {}).get("move", "freeze"), (lb or {}).get("move", "freeze")
    if cp["bow"] or lp["bow"]:
        return pose("bow")
    car_rests = cm in STILL and not (cb or {}).get("fig")
    lego_rests = lm in STILL and not (lb or {}).get("fig")
    if lego_rests and not car_rests:
        return cp
    if car_rests and not lego_rests:
        return lp
    if cp["arms"] == "side":                               # mid-turn: the whole body turns
        return cp
    return {**cp, "arms": lp["arms"]}


def figure_timeline(comp):
    """16th-note frames from the compiled score's blocks, plus the count-in and a ta-da."""
    from skills.matrix import digit_frame
    spb, count_in, beats = comp["spb"], comp["count_in"], comp["beats"]
    meter, dancers = comp.get("meter", 4), comp.get("dancers", 1)
    n16 = beats * 4
    car_t = _track(comp["blocks"], "car", n16, meter)
    lego_t = _track(comp["blocks"], "lego", n16, meter)
    out, last = [], None

    def add(beat, frame):
        nonlocal last
        if frame != last:
            out.append({"beat": round(beat, 4), "t": round((beat + count_in) * spb, 4),
                        "action": "frame", "params": {"frame": frame}})
            last = frame

    for i in range(count_in):
        add(-count_in + i, digit_frame(str(count_in - i)))
    for g in range(n16):
        f = [0] * (ROWS * COLS)
        car, lego = car_t[g], lego_t[g]
        if dancers == 2:
            mirror = bool(lego and lego[2]["mode"] == "mirror")
            draw(f, car[0] if car else STAND, DUO_X[0], DUO_BOUNDS[0], HIT if car and car[1] else HOLD)
            draw(f, lego[0] if lego else STAND, DUO_X[1], DUO_BOUNDS[1], HIT if lego and lego[1] else HOLD,
                 flip=mirror)
        else:
            hit = bool((car and car[1]) or (lego and lego[1]))
            mirror = bool(lego and lego[2]["mode"] == "mirror")
            draw(f, _solo_pose(car, lego), SOLO_X, (1, COLS - 2), HIT if hit else HOLD, flip_arms=mirror)
        if g % 4 == 0:                                     # the beat light, for one 16th
            down = (g // 4) % meter == 0
            for c in LIGHTS:
                for r in ((5, 6, 7) if down else (7,)):
                    f[r * COLS + c] = DOWNBEAT if down else BEAT
        add(g / 4, f)
    end = [0] * (ROWS * COLS)                              # ta-da
    for x0, bounds in (zip(DUO_X, DUO_BOUNDS) if dancers == 2 else [(SOLO_X, (0, COLS - 1))]):
        draw(end, pose("up/apart"), x0, bounds)
    add(beats, end)
    for i, k in enumerate(out):
        k["i"] = i
    return out
