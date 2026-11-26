"""The duet as ONE planner skill. The LLM writes a short score; deterministic
code compiles and performs it. Result feeds {dance.*} placeholders."""
import config
from dance.moves import CAR_MOVES, LEGO_MOVES, MODES
from skills.context import Cancelled, SkillError
from skills.registry import Arr, Enum, Int, Obj, Str, skill

SECTION = Obj({"bars": Int(1, config.MAX_SECTION_BARS), "mode": Enum(MODES),
               "car": Enum(CAR_MOVES), "lego": Enum(LEGO_MOVES)})


@skill("dance",
       "both robots dance; sets {dance.max_sync_error_ms}",
       args={"bpm": Int(*config.BPM_RANGE), "sections": Arr(SECTION, 1, config.MAX_SECTIONS)},
       sig=f"bpm {config.BPM_RANGE[0]}..{config.BPM_RANGE[1]}, sections: [{{bars 1..{config.MAX_SECTION_BARS}, "
           f"mode, car, lego}}]",
       ns="dance", motor=True, timeout_s=config.MAX_DANCE_S + 10,
       returns=["max_sync_error_ms", "mean_sync_error_ms", "completed"])
def dance(ctx, bpm, sections):
    if ctx.conductor is None:
        raise SkillError("no conductor")
    prev = ctx.status.get("state")
    ctx.status.set_state("DANCING", f"Dancing {sum(s['bars'] for s in sections)} bars at {bpm} bpm")
    try:
        r = ctx.conductor.perform({"bpm": bpm, "sections": sections}, cancel=ctx.cancel)
    finally:
        ctx.status.update(state=prev if prev != "DANCING" else "EXECUTING")
    if ctx.cancel.is_set():
        raise Cancelled("dance stopped")
    ctx.status.update(dance=r)
    return r


DANCE_CATALOG_EXTRA = (f" mode: {'|'.join(MODES)}\n"
                       f" car: {'|'.join(CAR_MOVES)}\n"
                       f" lego: {'|'.join(LEGO_MOVES)}")


@skill("dance_style", "dance in a named style (waltz, salsa, tango, west african...), choreographed from Wikipedia",
       args={"style": Str(40), "bars": Int(1, 8)}, ns="dance", motor=True, timeout_s=config.MAX_DANCE_S + 120,
       returns=["max_sync_error_ms", "mean_sync_error_ms", "completed"])
def dance_style(ctx, style, bars):
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
    score = ctx.choreograph(style, extract, bars)
    if not score:
        raise SkillError("I couldn't choreograph that style")
    why = (score.pop("why", "") or "").strip()
    if why and not why.endswith((".", "!", "?")):          # cut at a sentence or clause, never mid-word
        cut = max(why.rfind(". "), why.rfind(", "), why.rfind("; "))
        why = why[:cut] if cut > 20 else why.rsplit(" ", 1)[0]
    prev = ctx.status.get("state")
    ctx.status.set_state("DANCING", f"Dancing {style}: {sum(s['bars'] for s in score['sections'])} bars "
                                    f"at {score['bpm']} bpm")
    try:
        r = ctx.conductor.perform(score, cancel=ctx.cancel)
    finally:
        ctx.status.update(state=prev if prev != "DANCING" else "EXECUTING")
    if ctx.cancel.is_set():
        raise Cancelled("dance stopped")
    ctx.status.update(dance=r)
    modes = ", ".join(dict.fromkeys(s["mode"].replace("_", " ") for s in score["sections"]))
    source = (f" (from Wikipedia: {res['title']})" if res else
              " (Wikipedia was unreachable, so from the name alone)" if unreachable else
              " (no Wikipedia article found, so from the name alone)")
    text = (f"{style[:1].upper() + style[1:]}: {r['beats'] // 4} bars at {score['bpm']} bpm, {modes}. "
            + (why.rstrip(".") + ". " if why else "") + source.strip())
    topic = ctx.reply_to or config.TOPIC_REPLY
    ctx.publish(topic, text)
    ctx.published.append(topic)
    ctx.answers.append(text)
    return {**r, "style": style, "why": why, "source": res["title"] if res else None}
