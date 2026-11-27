"""Slots for the fast path (Part 2, A3): deterministic parsers, never trained.

Each intent's builder returns the exact canonical plan the LLM would have written, or None
when a required slot doesn't parse cleanly (no icon named, two icons named, no number for
a drive...). None sends the command to the LLM. Conventions follow the fine-tune's training
data (tools/finetune/gen_data.py), so the executor can't tell a fast plan from an LLM one.
"""
import re

from skills.font import ICON_NAMES

# ---------------------------------------------------------------- numbers
_NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
        "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "a hundred": 100, "one hundred": 100, "hundred": 100, "half a": 0.5, "a half": 0.5}


def words_to_digits(t):
    """'forty five degrees' -> '45 degrees', 'one hundred twenty bpm' -> '120 bpm', 'half a second' -> '0.5 second'."""
    t = re.sub(r"\bhalf an? (second|sec)\b", r"0.5 \1", t)
    t = re.sub(r"\b(one|a) hundred(?: and)? (twenty|thirty|forty|fifty|sixty)\b",
               lambda m: str(100 + _NUM[m.group(2)]), t)
    t = re.sub(r"\b(twenty|thirty|forty|fourty|fifty|sixty|seventy|eighty|ninety)[ -](one|two|three|four|five|six|seven|eight|nine)\b",
               lambda m: str(_NUM[m.group(1)] + _NUM[m.group(2)]), t)
    t = re.sub(r"\b(" + "|".join(sorted((k for k in _NUM if " " not in k and k != "hundred"), key=len, reverse=True)) + r")\b",
               lambda m: str(_NUM[m.group(1)]), t)
    t = re.sub(r"\b(a|one) hundred\b", "100", t)
    return t


def numbers(t):
    return [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", t)]


def _int(x):
    return int(x) if float(x).is_integer() else x


# ---------------------------------------------------------------- icons
ICON_WORDS = {
    "heart": ["heart", "love heart"], "smile": ["smile", "smiley", "happy face", "smiley face"],
    "check": ["check mark", "checkmark", "tick", "check"], "x": ["x mark", "cross", "an x", "x"],
    "star": ["star"], "note": ["music note", "musical note", "note"],
    "arrow_up": ["up arrow", "arrow pointing up"], "arrow_left": ["left arrow", "arrow pointing left"],
    "arrow_right": ["right arrow", "arrow pointing right"], "arrow_down": ["down arrow", "arrow pointing down"],
    "sad": ["sad face", "frown", "frowny face", "sad"], "question": ["question mark"],
    "building": ["building", "skyscraper", "tower"], "house": ["house", "home"],
    "tree": ["tree", "pine tree", "christmas tree"], "sun": ["sun", "sunshine"], "moon": ["moon", "crescent moon"],
    "cloud": ["cloud", "rain cloud"], "lightning": ["lightning bolt", "lightning", "thunderbolt"],
    "car": ["car"], "robot": ["robot face"], "rocket": ["rocket", "spaceship"], "flower": ["flower", "daisy"],
    "ghost": ["ghost"],
}


def find_icon(t):
    hits = set()
    for name, ws in ICON_WORDS.items():
        if name in ICON_NAMES and any(re.search(rf"\b{re.escape(w)}\b", t) for w in ws):
            hits.add(name)
    if "arrow" in " ".join(hits) or len(hits) > 1:      # "a left arrow" also contains nothing else; keep the arrow
        hits = {h for h in hits if h.startswith("arrow")} or hits
    return hits.pop() if len(hits) == 1 else None


# ---------------------------------------------------------------- builders
def step(skill, **args):
    return {"skill": skill, "args": args}


def b_show_icon(t, raw):
    name = find_icon(t)
    return [step("matrix_icon", name=name)] if name else None


_TEXT_VERB = r"(?:write|type|show|display|spell(?: out)?|scroll|put|print)"
_ON_DISPLAY = r"(?:\s+(?:on|across|to)\s+(?:the\s+)?(?:led\s+|l e d\s+)?(?:matrix|display|screen|leds?|lights))"


def b_write_text(t, raw):
    q = re.search(r"['\"“‘]([^'\"”’]{1,20})['\"”’]", raw)
    if q:
        return [step("matrix_text", text=q.group(1).strip())]
    m = re.search(rf"\b{_TEXT_VERB}\s+(?:the\s+(?:text|word)\s+)?([a-z0-9!?]+(?:\s+[a-z0-9!?]+){{0,2}}?){_ON_DISPLAY}?\s*[.!?]*\s*$",
                  t, re.I)
    if not m:
        return None
    words = m.group(1).split()
    if not words or words[0] in ("a", "an", "the", "me", "something", "anything", "it", "this", "that", "yourself",
                                 "myself", "you", "us", "them", "him", "her", "my", "your"):
        return None
    lo = m.start(1)
    original = raw[lo:lo + len(m.group(1))] if raw.lower()[lo:lo + len(m.group(1))] == m.group(1) else m.group(1)
    original = original.strip().rstrip("?!.,")
    if len(original.split()) != 1:
        return None                                    # "READY silently": unquoted text is one word; quote more
    return [step("matrix_text", text=original)]


def b_clear_matrix(t, raw):
    return [step("matrix_clear")]


def b_turn(t, raw):
    # Typos count as directions ("righht", "rght", "lefft"): defaulting to left on a misspelled
    # "right" would turn the wrong way.
    left = re.search(r"\b(left|lft|lef|lefft|leftt|letf|counter-?clockwise|anti-?clockwise)\b", t)
    right = (re.search(r"\b(r[ihg]{1,4}t|rite|clockwise)\b", t) and not re.search(r"\bcounter-?clockwise\b", t))
    if left and right:
        return None
    sign = -1 if right else 1                          # no direction = left (docs/MODEL.md)
    if re.search(r"\bquarter\b", t):
        deg = 90
    elif re.search(r"\b(half turn|halfway|turn around|about face|u-turn|other way)\b", t):
        deg = 180
    elif re.search(r"\b(full circle|all the way around|full 360|360)\b", t):
        deg = 360
    else:
        ns = [n for n in numbers(t) if 0 < n <= 360]
        if len(ns) != 1 or not re.search(r"\bdeg", t):
            return None
        deg = ns[0]
    return [step("turn_by", degrees=_int(sign * deg))]


_SPIN = re.compile(r"^(?:(?:can|could|would) you |only |just |go |please )*(?:spin|twirl|turn)(?: around| in place| in a circle)?"
                   r"(?: (once|twice|thrice|\d)(?: times?)?)?$|^(?:(?:can|could|would) you |only |just |please )*do (\d) spins?$")


def b_spin(t, raw):
    """Only phrasings it fully understands: a misspelled count ("thre times") must not become 1."""
    m = _SPIN.match(re.sub(r"[?!.]+$", "", core_slot_text(t)))
    if not m:
        return None
    w = m.group(1) or m.group(2) or "1"
    n = {"once": 1, "twice": 2, "thrice": 3}.get(w) or int(w)
    return [step("spin", times=n)] if n in (1, 2, 3) else None


def b_drive(t, raw):
    back = re.search(r"\b(back|backward|backwards|reverse)\b", t)
    fwd = re.search(r"\b(forward|forwards|ahead|straight)\b", t)
    if bool(back) == bool(fwd):
        return None
    ns = [n for n in numbers(t) if 0 < n <= 5]
    if len(ns) != 1 or not re.search(r"\b(s|sec|secs|second|seconds)\b", t):
        return None
    known_ly = {"slowly", "gently", "carefully", "quickly", "speedily", "only", "really", "early"}
    if any(w.endswith("ly") and w not in known_ly for w in re.findall(r"[a-z]+", t)):
        return None                                    # "quicly": a speed word we can't read
    if re.search(r"\b(slow|slowly|gently|carefully|crawl)\b", t):
        sp = 30
    elif re.search(r"\b(fast|quickly|full speed|speedily)\b", t):
        sp = 90
    else:
        sp = 60
    return [step("drive", speed=-sp if back else sp, seconds=_int(ns[0]))]


def b_stop(t, raw):
    return [step("stop")]


def b_dance_style(t, raw):
    from dance.styles import match_text
    p = match_text(raw)
    if not p:
        return None                                    # styles outside the library go to the LLM (Wikipedia)
    style = next(a for a in sorted(p["aliases"], key=len, reverse=True) if re.search(rf"\b{re.escape(a)}\b", t))
    bpm = re.search(r"(\d{2,3})\s*(bpm|beats)", t)
    bars = re.search(r"\b(\d)[ -]?bars?\b", t)
    on = re.findall(r"\b(?:on|with) (?:the |a )?([a-z]+)", t)
    known = {"robots", "robot", "matrix", "led", "leds", "screen", "display", "everything", "both", "stick", "lights"}
    if any(w not in known for w in on):
        return None                                    # "on eveyrthing": a place we can't read
    where = ("all" if re.search(r"\b(robots and (the )?(led )?matrix|everything|everywhere|both)\b", t) else
             "matrix" if re.search(r"\b(matrix|leds?|stick figure|screen|display)\b", t) else "robots")
    if bars and int(bars.group(1)) not in range(1, 9):
        return None
    return [step("dance_style", style=style, bars=int(bars.group(1)) if bars else 4,
                 bpm=int(bpm.group(1)) if bpm else 0, where=where)]


def _self(topic):
    return lambda t, raw: [step("self_report", topic=topic)]


def _lookup(what):
    return lambda t, raw: [step("lookup", what=what, about="")]


def b_memory_recall(t, raw):
    m = re.search(r"\b(?:what(?:'s| is| was)|do you (?:remember|know)|remind me(?: of)?|tell me)\s+my\s+([a-z' ]{2,30}?)"
                  r"(?:\s+(?:again|is|was|please|now))*\s*[?.!]*\s*$", t)
    if not m:
        return None
    key = m.group(1).strip()
    if len(key.split()) > 2:
        return None                                    # "my team reaal quick": leftover words we can't place
    return [step("memory", op="recall", key=key, value="")]


BUILD = {
    "show_icon": b_show_icon, "write_text": b_write_text, "clear_matrix": b_clear_matrix,
    "turn": b_turn, "spin": b_spin, "drive": b_drive, "stop": b_stop, "dance_style": b_dance_style,
    **{f"self_{k}": _self(k) for k in ("board", "pins", "devices", "software", "health", "network", "all")},
    **{f"lookup_{k}": _lookup(k) for k in ("time", "date", "weather", "forecast")},
    "memory_recall": b_memory_recall,
}


def core_slot_text(t):
    from fastpath.router import core_text
    return core_text(t)


def build(intent, text):
    """-> canonical steps, or None if a slot doesn't parse. Slots are read without the
    politeness around the command ("what's my lab partner again pls?" -> "lab partner")."""
    fn = BUILD.get(intent)
    if fn is None:
        return None
    core = core_slot_text(text.strip())
    t = words_to_digits(core.lower())
    return fn(t, core)
