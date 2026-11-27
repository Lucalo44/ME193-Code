"""Style library: what makes a waltz look like a waltz.

Each profile fixes a style's meter, tempo, rhythm and signature poses, so a well-known
style is danced the same recognisable way every time, and instantly (no LLM call). Styles
outside the library still go to the choreographer, grounded in Wikipedia.

A profile section is a normal score section (bars are filled in by score_for) plus two
optional keys the stick figures use:
  rhythm: {"car": "x...", "lego": "x.x."}     16th-note patterns (dance/rhythm.py)
  fig:    {"car": [...], "lego": [...]}       pose cycles, "arms/legs[/dx][/nohead]" (dance/figure.py)
The robots dance the section's car and lego moves; the figures dance the poses.

How many figures: two when the user asks for two (or names both robots), one when they ask
for one; otherwise two for partner dances and where="all", one for the rest.
"""
import re

import config
from dance.rhythm import FEELS, for_meter

SPIN = ["up/rise", "side/side", "up/rise/0/nohead", "side/side"]   # a turn, seen front-side-back-side

PROFILES = {
    "waltz": {
        "name": "Waltz", "aliases": ["waltz", "viennese waltz", "valse"], "bpm": 84, "meter": 3, "dancers": 2,
        "why": "3/4 time, down on 1 and rising on 2 and 3, partners mirroring in a closed hold",
        "sections": [
            {"mode": "mirror", "car": "step", "lego": "wave",
             "rhythm": {"car": "x...", "lego": "x..."},
             "fig": {"car": ["frame/plie", "frame/rise", "frame/rise"],
                     "lego": ["frame/plie", "frame/rise", "frame/rise"]}},
            {"mode": "mirror", "car": "shimmy", "lego": "wave",          # box step with a sway
             "rhythm": {"car": "x...", "lego": "x..."},
             "fig": {"car": ["frame/step_l/-1", "frame/rise", "frame/rise", "frame/step_r/1", "frame/rise", "frame/rise"],
                     "lego": ["frame/step_l/-1", "frame/rise", "frame/rise", "frame/step_r/1", "frame/rise", "frame/rise"]}},
            {"mode": "mirror", "car": "spin", "lego": "windmill",        # turning
             "rhythm": {"car": "x...", "lego": "x..."}, "fig": {"car": SPIN, "lego": SPIN}},
        ]},
    "salsa": {
        "name": "Salsa", "aliases": ["salsa", "mambo"], "bpm": 95, "meter": 4, "dancers": 2,
        "why": "quick-quick-slow on 1-2-3 and 5-6-7, holding 4 and 8, with hip action and cross-body turns",
        "sections": [
            {"mode": "mirror", "car": "step", "lego": "clap",
             "rhythm": {"car": "x.x.x...", "lego": "x.x.x..."},
             "fig": {"car": ["hips/step_l/-1", "hips/stand", "hips/stand/1", "hips/step_r/1", "hips/stand", "hips/stand/-1"],
                     "lego": ["hips/step_l/-1", "hips/stand", "hips/stand/1", "hips/step_r/1", "hips/stand", "hips/stand/-1"]}},
            {"mode": "call_response", "car": "shimmy", "lego": "clap",
             "rhythm": {"car": "x.x.x...", "lego": "x.x.x..."},
             "fig": {"car": ["up/apart/1", "up/apart/-1", "up/stand"], "lego": ["clap/plie", "out/stand", "clap/plie"]}},
            {"mode": "mirror", "car": "spin", "lego": "wave",
             "rhythm": {"car": "x.x.x...", "lego": "x.x.x..."}, "fig": {"car": SPIN, "lego": SPIN}},
        ]},
    "tango": {
        "name": "Tango", "aliases": ["tango", "argentine tango"], "bpm": 120, "meter": 4, "dancers": 2,
        "why": "slow, slow, quick-quick-slow: long walking steps and sharp head snaps in a close hold",
        "sections": [
            {"mode": "mirror", "car": "step", "lego": "point",
             "rhythm": {"car": "x.......x.......x...x...x.......", "lego": "x.......x.......x...x...x......."},
             "fig": {"car": ["frame/step_l", "frame/step_r", "frame/step_l", "frame/step_r", "frame/stand"],
                     "lego": ["frame/step_l", "frame/step_r", "frame/step_l", "frame/step_r", "frame/stand"]}},
            {"mode": "mirror", "car": "freeze", "lego": "point",          # the snap
             "rhythm": {"car": "x.......x...x...", "lego": "x.......x...x..."},
             "fig": {"car": ["frame/stand", "point/kick_r", "point/stand"], "lego": ["frame/stand", "point/kick_r", "point/stand"]}},
            {"mode": "mirror", "car": "spin", "lego": "windmill",
             "rhythm": {"car": "x...x...x.x.x...", "lego": "x...x...x.x.x..."}, "fig": {"car": SPIN, "lego": SPIN}},
        ]},
    "cha-cha": {
        "name": "Cha-cha", "aliases": ["cha cha cha", "cha-cha-cha", "cha cha", "cha-cha", "chacha"], "bpm": 120,
        "meter": 4, "dancers": 2,
        "why": "rock steps on 2 and 3, then the cha-cha-cha triple on 4-and-1",
        "sections": [
            {"mode": "mirror", "car": "step", "lego": "clap",
             "rhythm": {"car": "x...x...x...x.x.", "lego": "x...x...x...x.x."},
             "fig": {"car": ["hips/stand/1", "hips/step_l/-1", "hips/stand", "hips/step_r/1", "hips/stand/1"],
                     "lego": ["hips/stand/1", "hips/step_l/-1", "hips/stand", "hips/step_r/1", "hips/stand/1"]}},
            {"mode": "call_response", "car": "shimmy", "lego": "clap",
             "rhythm": {"car": "x...x...x...x.x.", "lego": "x...x...x...x.x."},
             "fig": {"car": ["up/apart/1", "up/apart/-1"], "lego": ["clap/plie", "out/stand"]}},
            {"mode": "mirror", "car": "spin", "lego": "pump",
             "rhythm": {"car": "x...x...x...x.x.", "lego": "x...x...x...x.x."}, "fig": {"car": SPIN, "lego": SPIN}},
        ]},
    "samba": {
        "name": "Samba", "aliases": ["samba"], "bpm": 100, "meter": 4, "dancers": 1,
        "why": "the samba bounce on 1-a-2, a 16th-note pulse with the knees, and quick turns",
        "sections": [
            {"mode": "unison", "car": "shimmy", "lego": "pump",
             "rhythm": {"car": "x..x", "lego": "x..x"},
             "fig": {"car": ["out/plie", "out/stand/1", "out/plie", "out/stand/-1"], "lego": ["up/plie", "down/stand"]}},
            {"mode": "canon", "car": "wiggle", "lego": "wave",
             "rhythm": {"car": "x..x", "lego": "x..x"},
             "fig": {"car": ["hips/step_l/-1", "hips/stand", "hips/step_r/1", "hips/stand"],
                     "lego": ["left_up/stand", "right_up/stand"]}},
            {"mode": "unison", "car": "spin", "lego": "windmill",
             "rhythm": {"car": "x..xx..xx.x.xxxx", "lego": "x..xx..xx.x.xxxx"}, "fig": {"car": SPIN, "lego": SPIN}},
        ]},
    "swing": {
        "name": "Swing", "aliases": ["swing", "lindy hop", "lindy", "jive", "jitterbug", "charleston", "boogie woogie"],
        "bpm": 140, "meter": 4, "dancers": 2,
        "why": "swung eighths, kicks and rock steps, and a partner swing-out",
        "sections": [
            {"mode": "mirror", "car": "step", "lego": "clap",
             "rhythm": {"car": "x..x", "lego": "x..x"},
             "fig": {"car": ["out/kick_r", "out/stand", "out/kick_l", "out/stand"],
                     "lego": ["out/kick_r", "out/stand", "out/kick_l", "out/stand"]}},
            {"mode": "call_response", "car": "shimmy", "lego": "clap",
             "rhythm": {"car": "x..x", "lego": "x..x"},
             "fig": {"car": ["up/apart/1", "up/apart/-1"], "lego": ["clap/plie", "out/stand"]}},
            {"mode": "mirror", "car": "spin", "lego": "wave",
             "rhythm": {"car": "x...x...x..xx...", "lego": "x...x...x..xx..."}, "fig": {"car": SPIN, "lego": SPIN}},
        ]},
    "ballet": {
        "name": "Ballet", "aliases": ["ballet", "ballerina", "classical dance"], "bpm": 72, "meter": 4, "dancers": 1,
        "why": "slow port de bras over plie and releve, a pirouette, and a held arabesque",
        "sections": [
            {"mode": "unison", "car": "step", "lego": "wave",
             "rhythm": {"car": "x.......", "lego": "x..."},
             "fig": {"car": ["down/plie", "down/rise"], "lego": ["down/stand", "out/stand", "fifth/stand", "out/stand"]}},
            {"mode": "solo_car", "car": "spin", "lego": "freeze",       # pirouette
             "rhythm": {"car": "x...x...x.x.xxxx"}, "fig": {"car": ["fifth/rise", "fifth/side", "fifth/rise/0/nohead", "fifth/side"]}},
            {"mode": "unison", "car": "freeze", "lego": "point",        # arabesque
             "rhythm": {"car": "x...............", "lego": "x.......x......."},
             "fig": {"car": ["out/kick_r"], "lego": ["out/kick_r", "point/kick_r"]}},
        ]},
    "hip hop": {
        "name": "Hip hop", "aliases": ["hip hop", "hip-hop", "hiphop", "street dance", "popping", "locking"],
        "bpm": 95, "meter": 4, "dancers": 1,
        "why": "a bounce on every beat under syncopated 16th-note arm hits, then a freeze",
        "sections": [
            {"mode": "unison", "car": "step", "lego": "pump",
             "rhythm": {"car": "x.x.", "lego": "x..x..x.x..x..x."},
             "fig": {"car": ["down/plie", "down/stand"], "lego": ["up/stand", "down/stand", "out/stand", "clap/stand"]}},
            {"mode": "unison", "car": "shimmy", "lego": "point",
             "rhythm": {"car": "x..x..x.", "lego": "x...x.x."},
             "fig": {"car": ["hips/apart/1", "hips/apart/-1"], "lego": ["point/stand", "left_up/stand"]}},
            {"mode": "unison", "car": "freeze", "lego": "freeze",
             "rhythm": {"car": "x..x..x.....x...", "lego": "x..x..x.....x..."},
             "fig": {"car": ["up/apart", "out/kick_l", "clap/plie", "point/apart"],
                     "lego": ["up/apart", "out/kick_l", "clap/plie", "point/apart"]}},
        ]},
    "breakdance": {
        "name": "Breakdance", "aliases": ["breakdance", "break dance", "breakdancing", "breaking", "b-boy", "bboy", "b-girl"],
        "bpm": 110, "meter": 4, "dancers": 1,
        "why": "toprock on the eighths, a 16th-note power move, then hit freezes",
        "sections": [
            {"mode": "unison", "car": "step", "lego": "clap",
             "rhythm": {"car": "x.x.", "lego": "x..."},
             "fig": {"car": ["hips/step_l/-1", "hips/stand", "hips/step_r/1", "hips/stand"], "lego": ["clap/stand", "out/stand"]}},
            {"mode": "unison", "car": "spin", "lego": "windmill",
             "rhythm": {"car": "xxxx", "lego": "xxxx"}, "fig": {"car": SPIN, "lego": SPIN}},
            {"mode": "unison", "car": "freeze", "lego": "freeze",
             "rhythm": {"car": "x.......x.x.....", "lego": "x.......x.x....."},
             "fig": {"car": ["point/kick_r", "up/apart", "out/kick_l"], "lego": ["point/kick_r", "up/apart", "out/kick_l"]}},
        ]},
    "disco": {
        "name": "Disco", "aliases": ["disco", "hustle", "funk"], "bpm": 120, "meter": 4, "dancers": 1,
        "why": "four on the floor, the up-and-down point, and hip rolls",
        "sections": [
            {"mode": "unison", "car": "step", "lego": "point",
             "rhythm": {"car": "x...", "lego": "x.x."},
             "fig": {"car": ["down/step_l/-1", "down/step_r/1"], "lego": ["right_up/stand", "down/stand"]}},
            {"mode": "unison", "car": "shimmy", "lego": "clap",
             "rhythm": {"car": "x.x.", "lego": "x..."},
             "fig": {"car": ["hips/apart/1", "hips/apart/-1"], "lego": ["clap/stand", "out/stand"]}},
            {"mode": "unison", "car": "spin", "lego": "pump",
             "rhythm": {"car": "x...", "lego": "x.x."}, "fig": {"car": SPIN, "lego": ["up/stand", "down/stand"]}},
        ]},
    "west african": {
        "name": "West African", "aliases": ["west african", "african", "afrobeat", "azonto", "kpanlogo"],
        "bpm": 120, "meter": 4, "dancers": 2,
        "why": "polyrhythm, feet on the quarters against arms on dotted eighths, and call and response",
        "sections": [
            {"mode": "call_response", "car": "step", "lego": "clap",
             "rhythm": {"car": "x...", "lego": "x..x..x..x..x..x"},
             "fig": {"car": ["down/plie", "up/stand"], "lego": ["clap/plie", "out/stand"]}},
            {"mode": "unison", "car": "shimmy", "lego": "pump",
             "rhythm": {"car": "x...", "lego": "x..x..x..x..x..x"},
             "fig": {"car": ["out/plie/1", "out/stand", "out/plie/-1", "out/stand"], "lego": ["up/plie", "down/stand"]}},
            {"mode": "canon", "car": "step", "lego": "wave",
             "rhythm": {"car": "x...", "lego": "x..x..x..x..x..x"},
             "fig": {"car": ["out/step_l", "out/step_r"], "lego": ["left_up/stand", "right_up/stand"]}},
        ]},
    "robot": {
        "name": "The robot", "aliases": ["robot", "the robot", "robotic", "robot dance"], "bpm": 100, "meter": 4,
        "dancers": 1,
        "why": "rigid right angles, every move landing dead on the beat and stopping",
        "sections": [
            {"mode": "unison", "car": "step", "lego": "point",
             "rhythm": {"car": "x...", "lego": "x..."},
             "fig": {"car": ["frame/stand", "robot_dn/stand"], "lego": ["frame/stand", "robot_dn/stand", "out/stand", "frame/stand"]}},
            {"mode": "call_response", "car": "spin", "lego": "pump",
             "rhythm": {"car": "x...", "lego": "x.x."},
             "fig": {"car": ["frame/stand", "side/side", "frame/stand/0/nohead", "side/side"],
                     "lego": ["frame/stand", "robot_dn/stand"]}},
            {"mode": "unison", "car": "freeze", "lego": "freeze",
             "rhythm": {"car": "x...x...x.x.x...", "lego": "x...x...x.x.x..."},
             "fig": {"car": ["frame/stand", "robot_dn/stand", "point/stand", "frame/stand"],
                     "lego": ["frame/stand", "robot_dn/stand", "point/stand", "frame/stand"]}},
        ]},
}
PARTNER_STYLES = {k for k, p in PROFILES.items() if p["dancers"] == 2}


def _norm(s):
    s = re.sub(r"[^a-z0-9 -]", " ", (s or "").lower())
    s = re.sub(r"dancers?\b", "dance", s)                  # "breakdancers" -> "breakdance"
    s = re.sub(r"\b(dance|dances|dancing|dancer|style|like|an?|the|traditional)\b", " ", s)
    return " ".join(s.split())


def match(style):
    """-> the profile for a style name ("a salsa", "Viennese waltz", "hip-hop dance"), or None."""
    s = _norm(style)
    if not s:
        return None
    best = None
    for key, p in PROFILES.items():
        for a in p["aliases"]:
            if re.search(rf"(^| ){re.escape(_norm(a) or a)}( |$)", s) and (best is None or len(a) > best[0]):
                best = (len(a), key)
    return {"key": best[1], **PROFILES[best[1]]} if best else None


# Words that mean something else in a command ("the robot spins", "swing your arms")
# only name a style in the user's words when "dance" follows ("swing dance", "do the robot").
_AMBIGUOUS = {"robot", "the robot", "robotic", "robot dance", "swing", "breaking", "funk", "locking", "popping",
              "hustle", "african"}


def match_text(text):
    """A library style named anywhere in the user's words ("loop two breakdancers on the
    matrix"), for when the planner wrote a plain dance. None if no style is named."""
    t = (text or "").lower()
    if re.search(r"\bdo the robot\b", t):
        return {"key": "robot", **PROFILES["robot"]}
    t = re.sub(r"dancers?\b", "dance", t)
    best = None
    for key, p in PROFILES.items():
        for a in p["aliases"]:
            pat = rf"\b{re.escape(a)}\b" + (r"\s*danc" if a in _AMBIGUOUS else "")
            if re.search(pat, t) and (best is None or len(a) > best[0]):
                best = (len(a), key)
    return {"key": best[1], **PROFILES[best[1]]} if best else None


def score_for(profile, bars, bpm=0):
    """Spread `bars` over the profile's sections (up to 4 bars each), at the user's tempo if given."""
    secs = profile["sections"]
    bars = max(1, min(len(secs) * config.MAX_SECTION_BARS, int(bars)))
    n = min(len(secs), bars, config.MAX_SECTIONS)
    split = [bars // n + (1 if i < bars % n else 0) for i in range(n)]
    out = [{**secs[i], "bars": b} for i, b in enumerate(split)]
    tempo = int(bpm) if bpm and bpm >= config.BPM_RANGE[0] else profile["bpm"]
    return {"bpm": tempo, "meter": profile["meter"], "style": profile["key"], "sections": out}


def apply_feel(score, feel, meter=4):
    """Choreographer path: turn its one-word feel into rhythms for every section."""
    if feel not in FEELS:
        return score
    car, lego = (for_meter(p, meter) for p in FEELS[feel])
    for s in score["sections"]:
        s.setdefault("rhythm", {"car": "" if s["car"] in ("freeze", "bow") else car,
                                "lego": "" if s["lego"] in ("freeze", "bow") else lego})
    return score


_TWO = re.compile(r"\b(two|2|both)\s+(\w+\s+){0,2}?(\w*dancers?|figures?|stick ?(figures?|men|people)|stickmen|people|"
                  r"persons|guys|performers|partners|bots|robots)\b|\b(duet|pair|couple|partners?|each other|"
                  r"one another|together)\b", re.I)
_ONE = re.compile(r"\b(a|one|1|single|the)\s+(\w+\s+)?(stick ?figure|figure|dancer|stick ?man)\b|"
                  r"\b(solo|alone|by (him|her|it)self)\b", re.I)


def dancers_for(text, where="matrix", default=1):
    """How many stick figures: the user's words first, then where="all" (one per robot), then the style."""
    t = text or ""
    if _TWO.search(t) or (re.search(r"\b(car|robot)\b", t, re.I) and re.search(r"\blego\b", t, re.I)):
        return 2
    if _ONE.search(t):
        return 1
    if where == "all":
        return 2
    return 2 if default == 2 else 1
