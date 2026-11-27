"""Dance score (what the LLM writes) -> one timeline per performer.

Score:
  {"bpm": 100, "sections": [{"bars": 2, "mode": "mirror", "car": "shimmy", "lego": "wave"}, ...]}
Optional (from the style library or the choreographer, never from the planner):
  score "meter" (3 or 4 beats per bar), "dancers" (1 or 2 stick figures), "style";
  section "rhythm" and "fig" per lane, for the stick figures (dance/figure.py).

Timeline keyframes carry both `beat` (relative to the first downbeat) and `t`
(seconds from the start of the count-in), so the conductor schedules in
seconds and the dashboard draws in beats.
"""
import config
from dance.moves import CAR_MOVES, LEGO_MOVES, MODES, Keyframe, car_move, lego_move
from dance.rhythm import valid as valid_rhythm

BEATS_PER_BAR = 4          # the default meter
LANES = ("car", "lego")


class ScoreError(ValueError):
    pass


def normalize(score):
    """Clamp a score into the legal range (the grammar already enforces enums)."""
    bpm = int(max(config.BPM_RANGE[0], min(config.BPM_RANGE[1], int(score.get("bpm", 100)))))
    meter = 3 if score.get("meter") == 3 else BEATS_PER_BAR
    secs = []
    for s in (score.get("sections") or [])[: config.MAX_SECTIONS]:
        if s.get("mode") not in MODES or s.get("car") not in CAR_MOVES or s.get("lego") not in LEGO_MOVES:
            raise ScoreError(f"bad section {s}")
        sec = {"bars": int(max(1, min(config.MAX_SECTION_BARS, int(s.get("bars", 1))))),
               "mode": s["mode"], "car": s["car"], "lego": s["lego"]}
        rhythm = {k: v for k, v in (s.get("rhythm") or {}).items() if k in LANES and valid_rhythm(v)}
        fig = {k: list(v)[:8] for k, v in (s.get("fig") or {}).items() if k in LANES and v}
        if rhythm:
            sec["rhythm"] = rhythm
        if fig:
            sec["fig"] = fig
        secs.append(sec)
    if not secs:
        raise ScoreError("no sections")
    # Respect the 60 s cap by dropping bars from the end.
    spb = 60.0 / bpm
    max_beats = int(config.MAX_DANCE_S / spb) - config.COUNT_IN_BEATS - 1
    total = 0
    out = []
    for s in secs:
        b = s["bars"] * meter
        if total + b > max_beats:
            bars = (max_beats - total) // meter
            if bars <= 0:
                break
            s = dict(s, bars=bars)
            b = bars * meter
        out.append(s)
        total += b
    norm = {"bpm": bpm, "sections": out}
    if meter != BEATS_PER_BAR:
        norm["meter"] = meter
    if score.get("dancers") == 2:
        norm["dancers"] = 2
    if score.get("style"):
        norm["style"] = str(score["style"])[:40]
    return norm


def compile_score(score, build=None, intensity=0.7, count_in=None):
    build = build or config.LEGO_BUILD
    score = normalize(score)
    meter = score.get("meter", BEATS_PER_BAR)
    # A bar of count-in: 4-3-2-1, or 3-2-1 for a waltz. 0 on a loop's later passes.
    ci = (config.COUNT_IN_BEATS if meter == BEATS_PER_BAR else meter) if count_in is None else count_in
    bpm = score["bpm"]
    spb = 60.0 / bpm
    car, lego, blocks = [], [], []
    start = 0
    for si, s in enumerate(score["sections"]):
        n = s["bars"] * meter
        mode, cm, lm = s["mode"], s["car"], s["lego"]

        def block(lane, name, st, beats, active):
            bl = {"lane": lane, "start": st, "beats": beats, "move": name, "mode": mode, "section": si,
                  "sec_start": start}
            if active:                     # the section's own move (not a rest inserted by the mode)
                for key in ("rhythm", "fig"):
                    if lane in s.get(key, {}):
                        bl[key] = s[key][lane]
            blocks.append(bl)

        def C(name, st, beats, mirror=False):
            car.extend(car_move(name, st, beats, intensity, mirror))
            block("car", name, st, beats, name == cm)

        def L(name, st, beats, mirror=False):
            lego.extend(lego_move(name, st, beats, intensity, mirror, build))
            block("lego", name, st, beats, name == lm)

        if mode == "unison":
            C(cm, start, n)
            L(lm, start, n)
        elif mode == "mirror":
            C(cm, start, n)
            L(lm, start, n, mirror=True)
        elif mode == "call_response":
            call = (meter + 1) // 2                     # 2 + 2 beats, or 2 + 1 in 3/4
            for bar in range(s["bars"]):
                b0 = start + bar * meter
                C(cm, b0, call)
                C("freeze", b0 + call, meter - call)
                L("freeze", b0, call)
                L(lm, b0 + call, meter - call)
        elif mode == "canon":
            C(cm, start, n)
            L("freeze", start, 1)
            L(lm, start + 1, n - 1)
        elif mode == "solo_car":
            C(cm, start, n)
            L("freeze", start, n)
        elif mode == "solo_lego":
            C("freeze", start, n)
            L(lm, start, n)
        start += n
    total_beats = start

    # Matrix: the count-in, a flash on every downbeat, a check at the end.
    mat = []
    for i in range(ci):
        mat.append(Keyframe(-ci + i, "matrix",
                            {"digit": str(ci - i)}, "count-in"))
    for b in range(0, total_beats, meter):
        mat.append(Keyframe(b, "matrix", {"icon": "note"}, "downbeat"))
        mat.append(Keyframe(b + 0.5, "matrix", {"clear": True}, "downbeat"))
    car.extend(mat)

    # Finish: everything home and still.
    car.append(Keyframe(total_beats, "wheels", {"l": 0, "r": 0}, "end"))
    car.append(Keyframe(total_beats, "matrix", {"icon": "check"}, "end"))
    if build == "wheels":
        lego.append(Keyframe(total_beats, "tank", {"l": 0, "r": 0}, "end"))
    else:
        lego.append(Keyframe(total_beats, "pose", {"a": 0, "b": 0, "speed": 50}, "end"))

    def finish(kfs):
        kfs.sort(key=lambda k: (k.t, k.action != "matrix"))
        out = []
        for i, k in enumerate(kfs):
            d = k.as_dict()
            d["i"] = i
            d["beat"] = d.pop("t")
            d["t"] = round((d["beat"] + ci) * spb, 4)
            out.append(d)
        return out

    car_t, lego_t = finish(car), finish(lego)
    seconds = (total_beats + ci) * spb
    if seconds > config.MAX_DANCE_S + 1e-6:
        raise ScoreError(f"dance too long: {seconds:.1f}s")
    return {"bpm": bpm, "spb": spb, "beats": total_beats, "count_in": ci, "meter": meter,
            "dancers": score.get("dancers", 1), "style": score.get("style"),
            "seconds": round(seconds, 3), "score": score, "blocks": blocks,
            "timelines": {"car": car_t, "lego": lego_t}}
