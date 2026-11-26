"""The duet as ONE planner skill. The LLM writes a short score; deterministic
code compiles and performs it. Result feeds {dance.*} placeholders."""
import re
import time

import config
from dance.moves import CAR_MOVES, LEGO_MOVES, MODES
from skills.context import Cancelled, SkillError
from skills.registry import Arr, Enum, Int, Obj, Str, skill

SECTION = Obj({"bars": Int(1, config.MAX_SECTION_BARS), "mode": Enum(MODES),
               "car": Enum(CAR_MOVES), "lego": Enum(LEGO_MOVES)})


WHERE = Enum(["robots", "matrix", "all"], default="robots")   # older plans without it still work

# Looping is read from the user's own words, not from a plan argument: the planner
# (and the fine-tuned model) only writes one phrase of 1-8 bars, and "loop it" is
# unambiguous enough that deterministic code shouldn't depend on the model noticing.
_LOOP = re.compile(r"\b(loop\w*|on repeat|repeat\w*|over and over|again and again|forever|non-?stop|"
                   r"continuous\w*|keep (dancing|going|it going)|until i (say )?stop|till i (say )?stop)\b", re.I)
_NUMWORD = {"once": 1, "twice": 2, "thrice": 3, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_COUNT = re.compile(r"\b(\d+|once|twice|thrice|two|three|four|five|six|seven|eight|nine|ten)\s*(times|x|loops|rounds|passes)?\b",
                    re.I)


def loop_count(text):
    """-> passes to play: 1 (no loop word), N ("loop it 3 times", "repeat twice"),
    or 0 = until STOP / MAX_LOOP_S ("loop a dance", "dance until I say stop")."""
    m = _LOOP.search(text or "")
    if not m:
        return 1
    # The count belongs to the loop only within the same clause ("loop the dance on the
    # matrix 2 times"), not to another action ("loop a dance, then spin 3 times").
    bounds = [0] + [x.end() for x in re.finditer(r"[.,;!?]|\bthen\b", text)] + [len(text)]
    lo = max(b for b in bounds if b <= m.start())
    hi = min([b for b in bounds if b > m.end()] or [len(text)])
    c = _COUNT.search(text[lo:hi])
    if c and (c.group(2) or c.group(1).lower() in ("once", "twice", "thrice")):
        n = c.group(1).lower()
        return max(1, min(config.MAX_LOOPS, int(n) if n.isdigit() else _NUMWORD[n]))
    return 0


def _loop_note(ctx):
    n = loop_count(ctx.question)
    return "" if n == 1 else f", looping {n} times" if n else ", looping until STOP"


def perform_looped(ctx, score, where):
    """Play the score once, or loop it (no count-in after the first pass) until the
    requested passes, STOP, or MAX_LOOP_S. Returns the last pass's report + loop info."""
    passes = loop_count(ctx.question)
    t0, done, r = time.time(), 0, None
    while True:
        r = ctx.conductor.perform(score, cancel=ctx.cancel, where=where, count_in=None if done == 0 else 0)
        done += 1
        if ctx.cancel.is_set() or (passes and done >= passes):
            break
        if time.time() - t0 + r.get("seconds", 0) > config.MAX_LOOP_S:
            break
    return {**r, "loops": done, "looped": passes != 1}


@skill("dance",
       "dance; where: robots, matrix (a stick figure on the LED matrix) or all; sets {dance.max_sync_error_ms}",
       args={"bpm": Int(*config.BPM_RANGE), "sections": Arr(SECTION, 1, config.MAX_SECTIONS), "where": WHERE},
       sig=f"bpm {config.BPM_RANGE[0]}..{config.BPM_RANGE[1]}, sections: [{{bars 1..{config.MAX_SECTION_BARS}, "
           f"mode, car, lego}}], where: robots|matrix|all",
       ns="dance", motor=True, timeout_s=config.MAX_LOOP_S + 10,
       returns=["max_sync_error_ms", "mean_sync_error_ms", "completed"])
def dance(ctx, bpm, sections, where):
    if ctx.conductor is None:
        raise SkillError("no conductor")
    prev = ctx.status.get("state")
    ctx.status.set_state("DANCING", f"Dancing {sum(s['bars'] for s in sections)} bars at {bpm} bpm"
                                    + (" on the LED matrix" if where == "matrix" else "") + _loop_note(ctx))
    try:
        r = perform_looped(ctx, {"bpm": bpm, "sections": sections}, where)
    finally:
        ctx.status.update(state=prev if prev != "DANCING" else "EXECUTING")
    if ctx.cancel.is_set():
        raise Cancelled("dance stopped")
    ctx.status.update(dance=r)
    return r


DANCE_CATALOG_EXTRA = (f" mode: {'|'.join(MODES)}\n"
                       f" car: {'|'.join(CAR_MOVES)}\n"
                       f" lego: {'|'.join(LEGO_MOVES)}")


@skill("dance_style", "dance in a named style (waltz, salsa, ballet, west african...), choreographed from Wikipedia; "
       "bpm 0 = the style's own tempo; where: robots|matrix|all",
       args={"style": Str(40), "bars": Int(1, 8), "bpm": Int(0, config.BPM_RANGE[1], default=0), "where": WHERE},
       sig="style, bars 1..8, bpm 0..160, where: robots|matrix|all",
       ns="dance", motor=True, timeout_s=config.MAX_LOOP_S + 120,
       returns=["max_sync_error_ms", "mean_sync_error_ms", "completed"])
def dance_style(ctx, style, bars, bpm, where):
    """Grounded choreography: read what Wikipedia says about the style, let the LLM
    turn that description into a score (grammar-constrained), then perform it."""
    if ctx.conductor is None or ctx.choreograph is None:
        raise SkillError("no conductor or choreographer")
    q = style if "dance" in style.lower() else f"{style} dance"
    res, unreachable = None, False
    if ctx.wiki is not None:
        try:
            res = ctx.wiki.lookup(q)
        except Exception:  # noqa: BLE001  dance anyway, from the style name alone
            res, unreachable = None, True
    extract = res["extract"] if res else ""
    bpm = bpm if bpm >= config.BPM_RANGE[0] else 0          # 0 (or nonsense) = let the style decide
    score = ctx.choreograph(style, extract, bars, bpm)
    if not score:
        raise SkillError("I couldn't choreograph that style")
    why = (score.pop("why", "") or "").strip()
    if why and not why.endswith((".", "!", "?")):          # cut at a sentence or clause, never mid-word
        cut = max(why.rfind(". "), why.rfind(", "), why.rfind("; "))
        why = why[:cut] if cut > 20 else why.rsplit(" ", 1)[0]
        if len(why.split()) < 6:                           # "a simple, abstract" says nothing: drop it
            why = ""
    prev = ctx.status.get("state")
    ctx.status.set_state("DANCING", f"Dancing {style}: {sum(s['bars'] for s in score['sections'])} bars "
                                    f"at {score['bpm']} bpm" + (" on the LED matrix" if where == "matrix" else "")
                                    + _loop_note(ctx))
    try:
        r = perform_looped(ctx, score, where)
    finally:
        ctx.status.update(state=prev if prev != "DANCING" else "EXECUTING")
    if ctx.cancel.is_set():
        raise Cancelled("dance stopped")
    ctx.status.update(dance=r)
    # On the matrix there is no car or LEGO bot: the figure's legs follow the car lane
    # and its arms the LEGO lane, so describe the sections in those terms.
    FIGURE_WORDS = {"unison": "arms and legs together", "mirror": "mirrored arms",
                    "call_response": "legs calling, arms answering", "canon": "arms a beat behind the legs",
                    "solo_car": "footwork", "solo_lego": "arm work"}
    words = (lambda m: FIGURE_WORDS[m]) if where == "matrix" else (lambda m: m.replace("_", " "))
    modes = ", ".join(dict.fromkeys(words(s["mode"]) for s in score["sections"]))
    source = (f" (from Wikipedia: {res['title']})" if res else
              " (Wikipedia was unreachable, so from the name alone)" if unreachable else
              " (no Wikipedia article found, so from the name alone)")
    on = {"matrix": " on the LED matrix", "all": " with the robots and the LED matrix"}.get(where, "")
    loops = (f", looped {r['loops']} times" if r.get("looped") and r["loops"] > 1 else "")
    text = (f"{style[:1].upper() + style[1:]}{on}: {r['beats'] // 4} bars at {score['bpm']} bpm{loops}, {modes}. "
            + (why.rstrip(".") + ". " if why else "") + source.strip())
    topic = ctx.reply_to or config.TOPIC_REPLY
    ctx.publish(topic, text)
    ctx.published.append(topic)
    ctx.answers.append(text)
    return {**r, "style": style, "why": why, "source": res["title"] if res else None}
