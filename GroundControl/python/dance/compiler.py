"""Dance score (what the LLM writes) -> one timeline per performer.

Score:
  {"bpm": 100, "sections": [{"bars": 2, "mode": "mirror", "car": "shimmy", "lego": "wave"}, ...]}

Timeline keyframes carry both `beat` (relative to the first downbeat) and `t`
(seconds from the start of the count-in), so the conductor schedules in
seconds and the dashboard draws in beats.
"""
import config
from dance.moves import CAR_MOVES, LEGO_MOVES, MODES, Keyframe, car_move, lego_move

BEATS_PER_BAR = 4


class ScoreError(ValueError):
    pass


def normalize(score):
    """Clamp a score into the legal range (the grammar already enforces enums)."""
    bpm = int(max(config.BPM_RANGE[0], min(config.BPM_RANGE[1], int(score.get("bpm", 100)))))
    secs = []
    for s in (score.get("sections") or [])[: config.MAX_SECTIONS]:
        if s.get("mode") not in MODES or s.get("car") not in CAR_MOVES or s.get("lego") not in LEGO_MOVES:
            raise ScoreError(f"bad section {s}")
        secs.append({"bars": int(max(1, min(config.MAX_SECTION_BARS, int(s.get("bars", 1))))),
                     "mode": s["mode"], "car": s["car"], "lego": s["lego"]})
    if not secs:
        raise ScoreError("no sections")
    # Respect the 60 s cap by dropping bars from the end.
    spb = 60.0 / bpm
    max_beats = int(config.MAX_DANCE_S / spb) - config.COUNT_IN_BEATS - 1
    total = 0
    out = []
    for s in secs:
        b = s["bars"] * BEATS_PER_BAR
        if total + b > max_beats:
            bars = (max_beats - total) // BEATS_PER_BAR
            if bars <= 0:
                break
            s = dict(s, bars=bars)
            b = bars * BEATS_PER_BAR
        out.append(s)
        total += b
    return {"bpm": bpm, "sections": out}


def compile_score(score, build=None, intensity=0.7):
    build = build or config.LEGO_BUILD
    score = normalize(score)
    bpm = score["bpm"]
    spb = 60.0 / bpm
    car, lego, blocks = [], [], []
    start = 0
    for si, s in enumerate(score["sections"]):
        n = s["bars"] * BEATS_PER_BAR
        mode, cm, lm = s["mode"], s["car"], s["lego"]

        def C(name, st, beats, mirror=False):
            car.extend(car_move(name, st, beats, intensity, mirror))
            blocks.append({"lane": "car", "start": st, "beats": beats, "move": name, "mode": mode, "section": si})

        def L(name, st, beats, mirror=False):
            lego.extend(lego_move(name, st, beats, intensity, mirror, build))
            blocks.append({"lane": "lego", "start": st, "beats": beats, "move": name, "mode": mode, "section": si})

        if mode == "unison":
            C(cm, start, n)
            L(lm, start, n)
        elif mode == "mirror":
            C(cm, start, n)
            L(lm, start, n, mirror=True)
        elif mode == "call_response":
            for bar in range(s["bars"]):
                b0 = start + bar * BEATS_PER_BAR
                C(cm, b0, 2)
                C("freeze", b0 + 2, 2)
                L("freeze", b0, 2)
                L(lm, b0 + 2, 2)
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

    # Matrix: count-in 3-2-1, a flash on every downbeat, a check at the end.
    mat = []
    for i in range(config.COUNT_IN_BEATS):
        mat.append(Keyframe(-config.COUNT_IN_BEATS + i, "matrix",
                            {"digit": str(config.COUNT_IN_BEATS - i)}, "count-in"))
    for b in range(0, total_beats, BEATS_PER_BAR):
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
            d["t"] = round((d["beat"] + config.COUNT_IN_BEATS) * spb, 4)
            out.append(d)
        return out

    car_t, lego_t = finish(car), finish(lego)
    seconds = (total_beats + config.COUNT_IN_BEATS) * spb
    if seconds > config.MAX_DANCE_S + 1e-6:
        raise ScoreError(f"dance too long: {seconds:.1f}s")
    return {"bpm": bpm, "spb": spb, "beats": total_beats, "count_in": config.COUNT_IN_BEATS,
            "seconds": round(seconds, 3), "score": score, "blocks": blocks,
            "timelines": {"car": car_t, "lego": lego_t}}
