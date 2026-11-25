"""The duet as ONE planner skill. The LLM writes a short score; deterministic
code compiles and performs it. Result feeds {dance.*} placeholders."""
import config
from dance.moves import CAR_MOVES, LEGO_MOVES, MODES
from skills.context import Cancelled, SkillError
from skills.registry import Arr, Enum, Int, Obj, skill

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
