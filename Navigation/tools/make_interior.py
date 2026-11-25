"""Build web/interior.js: the two floors of the Tsungming Tu Complex the route uses, traced from
photos of the floor plans posted in the building (data/floorplans/level1.jpg and level0.jpg).

    python3 tools/make_interior.py            # needs opencv-python-headless
    python3 tools/make_interior.py --debug    # also writes data/floorplans/debug_level*.png

Each photo is traced by colour (teal = rooms, pale = corridors, red = the cafe), then scaled,
turned and moved onto the outdoor map in web/map.js (metres east/north of its origin).
How the photos are lined up (all measured by hand, good to a couple of metres):
  * "up" on both plans runs along Anderson Hall's long walls
  * level 1: 41 px per metre (the 32-step first corridor is 950 px); the corner where Anderson
    meets the newer wing is pinned to the same corner of the building outline
  * level 0: 31.5 px per metre (the 13-step corridor after the stairs is 298 px); pinned to
    level 1 through the North Stair, which is on both photos
ROUTE is the walked path, in photo pixels, with the step count at each point (medians of the
cleaned demo walks), so the phone page can put you on the plan from your step count.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
PLANS = HERE / "data" / "floorplans"
JUNCTION_GUESS = (-5.8, 19.9)          # Anderson/new-wing corner in map.js metres (east, north)
LEVELS = {
    "1": {"file": "level1.jpg", "px_per_m": 41.0, "pin_px": (770, 1145), "outside_px": [(766, 0, 1125, 1140)], "stairs_px": [(485, 1630, 575, 1800)],
          "labels": [("109G", 235, 168), ("109F", 237, 270), ("109E", 242, 407), ("109A", 22, 68), ("109B", 24, 213),
                     ("109C", 26, 323), ("109D", 28, 433), ("Nelson Auditorium 112", 583, 180), ("113C", 473, 466),
                     ("113B", 645, 440), ("113A", 697, 567), ("113D", 477, 611), ("113", 630, 742),
                     ("Burdon Lounge 108", 143, 655), ("105A", 97, 907), ("106", 271, 905),
                     ("Dean's Office 105", 223, 1108), ("105C", 55, 1226), ("105D", 201, 1280), ("102", 577, 1246),
                     ("116A", 715, 1226), ("103", 717, 1418), ("EPDC 116", 1000, 1345), ("Conference 117", 1016, 1708),
                     ("Mech. Eng. Office 150", 562, 1900), ("174", 211, 1932), ("North Stair", 530, 1600),
                     ("Restroom", 675, 915), ("Restroom", 678, 1080)]},
    "0": {"file": "level0.jpg", "px_per_m": 31.5, "stairs_px": [(340, 1432, 510, 1490), (200, 1466, 285, 1494)],
          "labels": [("Teaching Lab 014", 58, 220), ("Research Lab 016", 397, 327), ("Teaching Lab 013", 66, 488),
                     ("012", 20, 625), ("010", 135, 625), ("016D", 318, 632), ("016B", 446, 632), ("016C", 505, 632),
                     ("Research Lab 007", 78, 806), ("Teaching Lab 019", 405, 785), ("Teaching Lab 006", 60, 1030),
                     ("Lab 020", 421, 1075), ("Lab 020D", 543, 1073), ("Lab 022A", 336, 1250),
                     ("Blake-Pearlman Lab 022", 512, 1262), ("Loading 024", 926, 1143), ("003", 116, 1320),
                     ("001", 108, 1441), ("North Stair", 430, 1412), ("Kindlevan Cafe", 885, 1440),
                     ("Atrium", 568, 1808), ("Nolop Makerspace 066", 147, 1910), ("Teaching Lab 029A", 895, 1873)]},
}
STAIR_MATCH = {"1": (485, 1715), "0": (340, 1460)}      # west end of the North Stair on each photo
# the walk: (level, x px, y px, step count there, what happens there)
ROUTE = [("1", 365, 775, 0, "start"), ("1", 378, 1725, 32.5, "left"), ("1", 485, 1725, 34, "stairs"),
         ("0", 510, 1460, 56.5, "stairs end"), ("0", 560, 1460, 60, "left"), ("0", 560, 1405, 63.5, "left"),
         ("0", 264, 1405, 76.5, "left"), ("0", 262, 1580, 88.5, "arrive")]
DESTINATION = ("0", 147, 1910)                          # a point inside the Makerspace, to highlight that room


def trace(img, level):
    """Colour masks -> simplified outlines, in photo pixels."""
    b, g, r = [img[:, :, i].astype(int) for i in range(3)]
    ink = ((np.maximum(np.maximum(r, g), b) < 60).astype(np.uint8)) * 255       # the hand-drawn route
    ink = cv2.dilate(cv2.erode(ink, np.ones((7, 7), np.uint8)), np.ones((17, 17), np.uint8))
    img = cv2.inpaint(img, ink, 9, cv2.INPAINT_TELEA)
    b, g, r = [img[:, :, i].astype(int) for i in range(3)]
    teal = ((r < 70) & (b > 95) & (g > 80)).astype(np.uint8) * 255
    red = ((r > 140) & (g < 115) & (b < 105)).astype(np.uint8) * 255
    if level == "1":
        pale = ((r > 165) & (g > 165) & (r - b < 32)).astype(np.uint8) * 255   # grey corridor, not the brown wall
    else:
        pale = ((b - r > 4) & (g > 120) & (r > 90)).astype(np.uint8) * 255     # blue-grey corridor, not the warm wall
    k = np.ones((5, 5), np.uint8)
    teal = cv2.morphologyEx(teal, cv2.MORPH_OPEN, k)
    slab = cv2.morphologyEx(cv2.bitwise_or(cv2.bitwise_or(teal, pale), red), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    slab = cv2.morphologyEx(slab, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))

    def outlines(mask, min_area, eps):
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist() for c in cs if cv2.contourArea(c) >= min_area]
    for x0, y0, x1, y1 in LEVELS[level].get("outside_px", []):               # glare the colour test lets through
        slab[y0:y1, x0:x1] = 0
        red[y0:y1, x0:x1] = 0
    cs, _ = cv2.findContours(cv2.morphologyEx(red, cv2.MORPH_OPEN, k), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cafe = [cv2.convexHull(c).reshape(-1, 2).tolist() for c in cs if cv2.contourArea(c) >= 6000]
    cafe = [cv2.approxPolyDP(np.array(c), 6, True).reshape(-1, 2).tolist() for c in cafe]
    return {"slab": outlines(slab, 40000, 5), "rooms": outlines(teal, 900, 4), "cafe": cafe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    campus = json.loads((HERE / "web" / "map.js").read_text().split("window.CAMPUS = ")[1].rstrip(";\n"))
    anderson = next(b["p"] for b in campus["buildings"] if b.get("n") == "Anderson")
    pts = [(anderson[i], anderson[i + 1]) for i in range(0, len(anderson), 2)]
    edges = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    p, q = max(edges, key=lambda e: math.dist(*e))                      # the long west wall
    up = math.atan2(q[0] - p[0], q[1] - p[1])                           # compass bearing of "up" on the plans
    if math.cos(up) < 0:
        up += math.pi                                                   # the plans have north-ish at the top
    pin = min(pts, key=lambda v: math.dist(v, JUNCTION_GUESS))
    ux, uy, rx, ry = math.sin(up), math.cos(up), math.cos(up), -math.sin(up)

    def to_world(level, origin_px, origin_w):
        s = LEVELS[level]["px_per_m"]
        def f(x, y):
            dr, du = (x - origin_px[0]) / s, -(y - origin_px[1]) / s
            return [round(origin_w[0] + dr * rx + du * ux, 2), round(origin_w[1] + dr * ry + du * uy, 2)]
        return f
    w1 = to_world("1", LEVELS["1"]["pin_px"], pin)
    w0 = to_world("0", STAIR_MATCH["0"], w1(*STAIR_MATCH["1"]))
    W = {"1": w1, "0": w0}

    def is_dest(level, poly):
        return level == DESTINATION[0] and cv2.pointPolygonTest(np.array(poly), DESTINATION[1:], False) > 0

    out = {"up": round(math.degrees(up), 1), "floors": {}, "route": []}
    for level, L in LEVELS.items():
        img = cv2.imread(str(PLANS / L["file"]))
        t = trace(img, level)
        flat = lambda poly: [c for x, y in poly for c in W[level](x, y)]
        out["floors"][level] = {
            "slab": [flat(p) for p in t["slab"]], "cafe": [flat(p) for p in t["cafe"]],
            "rooms": [flat(p) for p in t["rooms"] if not is_dest(level, p)],
            "dest": [flat(p) for p in t["rooms"] if is_dest(level, p)],
            "stairs": [flat([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) for x0, y0, x1, y1 in L["stairs_px"]],
            "labels": [{"n": n, "x": W[level](x, y)[0], "y": W[level](x, y)[1]} for n, x, y in L["labels"]]}
        print(f"level {level}: {len(t['slab'])} floor outline(s), {len(t['rooms'])} rooms, {len(t['cafe'])} cafe")
        if a.debug:
            dbg = (img * 0.35).astype(np.uint8)
            for p_ in t["slab"]:
                cv2.polylines(dbg, [np.array(p_)], True, (255, 255, 255), 3)
            for p_ in t["rooms"]:
                cv2.polylines(dbg, [np.array(p_)], True, (154, 61, 255), 2)
            for p_ in t["cafe"]:
                cv2.polylines(dbg, [np.array(p_)], True, (61, 154, 255), 3)
            r_ = [(x, y) for lv, x, y, _, _ in ROUTE if lv == level]
            cv2.polylines(dbg, [np.array(r_)], False, (208, 229, 44), 4)
            cv2.imwrite(str(PLANS / f"debug_level{level}.png"), dbg)
    for level, x, y, step, what in ROUTE:
        wx, wy = W[level](x, y)
        out["route"].append({"x": wx, "y": wy, "floor": level, "step": step, "what": what})
    dest = HERE / "web" / "interior.js"
    dest.write_text("// Made by tools/make_interior.py from photos of the building's posted floor plans.\n"
                    "window.INTERIOR = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"wrote {dest} ({dest.stat().st_size // 1024} kB); plan 'up' is compass {out['up']} deg")
    legs = [math.dist((a_["x"], a_["y"]), (b_["x"], b_["y"])) for a_, b_ in zip(out["route"], out["route"][1:])]
    print("route legs (m):", [round(v, 1) for v in legs], "total", round(sum(legs), 1))


if __name__ == "__main__":
    main()
