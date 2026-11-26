"""A stick figure that dances on the 8x13 LED matrix.

The figure is a third performer. Its legs and body follow the CAR's move, and its
arms follow the LEGO bot's move. So on its own (where="matrix") it dances both
parts, and with the robots (where="all") it mirrors them. Frames are built from
layers (head, torso, arms, legs) and shifted sideways for sways, so a dozen
pieces of pixel art cover every move combination.

    figure_timeline(compiled_score) -> [{"t", "beat", "action": "frame", "params": {"frame": [...104]}}]
"""
from hal import COLS, ROWS

W = 7                       # figure window width, centred on the matrix
X0 = (COLS - W) // 2        # = 3

# Each layer: {row: "7-char string"}; '#' = lit. Row 0 is the top.
HEAD = {0: "...#..."}
TORSO = {2: "...#...", 3: "...#...", 4: "...#..."}
ARMS = {
    "down":     {1: "..###..", 2: ".#...#.", 3: ".#...#."},
    "up":       {0: ".#.#.#.", 1: "..###.."},
    "out":      {1: "#######"},
    "left_up":  {0: ".#.#...", 1: "..###..", 2: ".....#."},
    "right_up": {0: "...#.#.", 1: "..###..", 2: ".#....."},
    "clap":     {0: "..###..", 1: "..###.."},
    "point":    {1: "..#####", 2: ".#....."},
    "side":     {1: "...##..", 2: "....#.."},          # seen from the side (spins)
}
LEGS = {
    "stand":   {5: "..#.#..", 6: "..#.#..", 7: "..#.#.."},
    "apart":   {5: "..#.#..", 6: ".#...#.", 7: "#.....#"},
    "step_l":  {5: "..#.#..", 6: ".#..#..", 7: "#...#.."},
    "step_r":  {5: "..#.#..", 6: "..#..#.", 7: "..#...#"},
    "plie":    {5: ".#...#.", 6: ".#...#.", 7: ".#...#."},
    "side":    {5: "...#...", 6: "...#...", 7: "..##..."},
}
BOW = {3: "..####.", 4: "..#.#..", 5: "..#.#..", 6: "..#.#..", 7: "..#.#.."}   # bent forward, arm hanging


def render(arms="down", legs="stand", dx=0, bow=False, head=True):
    f = [0] * (ROWS * COLS)
    layers = [BOW] if bow else [HEAD if head else {}, TORSO, ARMS[arms], LEGS[legs]]
    for layer in layers:
        for r, row in layer.items():
            for i, ch in enumerate(row):
                c = X0 + i + dx
                if ch == "#" and 0 <= c < COLS:
                    f[r * COLS + c] = 7
    return f


def _legs(move, phase):
    """Body/legs pose for a CAR move at `phase` beats into it -> (legs, dx, special)."""
    q = int(phase * 4)                       # quarter-beat index
    if move == "shimmy":
        return "apart", (1 if q % 2 == 0 else -1), None
    if move == "spin":
        return {0: ("apart", 0, None), 1: ("side", 0, "side"), 2: ("apart", 0, "back"), 3: ("side", 0, "side")}[q % 4]
    if move == "step":
        return ("step_l" if int(phase) % 2 == 0 else "step_r"), 0, None
    if move == "wiggle":
        return "stand", (1 if q % 2 == 0 else -1), None
    if move == "bow":
        return "stand", 0, "bow"
    return "stand", 0, None                  # freeze


def _arms(move, phase, mirror=False):
    """Arm pose for a LEGO move at `phase` beats into it."""
    q = int(phase * 4)
    half = int(phase * 2) % 2
    flip = {"left_up": "right_up", "right_up": "left_up"}
    pick = {
        "wave": "left_up" if half == 0 else "down",
        "clap": "clap" if half == 0 else "out",
        "pump": "up" if half == 0 else "down",
        "point": "point",
        "windmill": ["left_up", "out", "right_up", "down"][q % 4],
        "freeze": "down",
        "bow": "down",
    }.get(move, "down")
    return flip.get(pick, pick) if mirror else pick


def figure_timeline(comp):
    """Quarter-beat frames from the compiled score's blocks (+ count-in and ending)."""
    from skills.matrix import digit_frame
    spb, count_in, beats = comp["spb"], comp["count_in"], comp["beats"]
    lanes = {"car": [], "lego": []}
    for bl in comp["blocks"]:
        lanes[bl["lane"]].append(bl)

    def at(lane, b):
        for bl in lanes[lane]:
            if bl["start"] <= b < bl["start"] + bl["beats"]:
                return bl
        return None

    out, last = [], None

    def add(beat, frame):
        nonlocal last
        if frame != last:
            out.append({"beat": round(beat, 4), "t": round((beat + count_in) * spb, 4),
                        "action": "frame", "params": {"frame": frame}})
            last = frame

    for i in range(count_in):
        add(-count_in + i, digit_frame(str(count_in - i)))
    for q in range(beats * 4):
        b = q / 4
        car, lego = at("car", b), at("lego", b)
        legs, dx, special = _legs(car["move"], b - car["start"]) if car else ("stand", 0, None)
        arms = _arms(lego["move"], b - lego["start"], mirror=bool(lego and lego["mode"] == "mirror")) if lego else "down"
        if special == "bow" or (lego and lego["move"] == "bow" and (b - lego["start"]) < 1.5):
            frame = render(bow=True)
        elif special == "side":
            frame = render(arms="side", legs="side")
        elif special == "back":
            frame = render(arms=arms, legs=legs, head=False)
        else:
            frame = render(arms=arms, legs=legs, dx=dx)
        add(b, frame)
    add(beats, render(arms="up", legs="apart"))          # ta-da
    for i, k in enumerate(out):
        k["i"] = i
    return out
