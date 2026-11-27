#!/usr/bin/env python3
"""Train and calibrate the fast-path intent classifier (Part 2, A3), on the laptop.

    venv/bin/python tools/fastpath/train.py

1. Data: the fine-tune's synthetic commands (tools/finetune/data, already free of eval
   commands and their near-duplicates). A plan with ONE whitelisted step gets that intent;
   everything else (multi-step, refusals, answers, math, Wikipedia, dance scores...) is
   "none". Extra hard negatives are generated from the positives: "X and send done",
   "don't X", "X then turn left", "can you X?" questions about it.
2. Train: python/fastpath/model.py (numpy), Adam, cross-entropy.
3. Calibrate: on val.jsonl (never trained on), pick the (tau, margin) with the most
   coverage at >= 99.5% precision, where precision counts a fast plan as right only if it
   equals the plan in the data (drive speed compared as slow / normal / fast).
4. Report: the same on tools/eval/commands.yaml (held out, scored like run_eval) and
   tools/eval/dev.yaml. Writes python/fastpath/model.npz, calibration.json and
   docs/bench/fastpath.json.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import config  # noqa: E402,F401
import skills  # noqa: E402,F401
from fastpath.model import INTENTS, Adam, Net, features  # noqa: E402
from fastpath.router import Router  # noqa: E402
from planner.normalize import normalize  # noqa: E402

DATA = ROOT / "tools" / "finetune" / "data"
OUT = ROOT / "python" / "fastpath"
R = random.Random(193)
PRECISION_TARGET = 0.999   # the plan asks for 99.5%; intents without slots (time, self-reports) have only the net to catch a wrong guess


def intent_of(steps):
    if len(steps) != 1:
        return "none"
    s, a = steps[0]["skill"], steps[0]["args"]
    if s == "matrix_icon":
        return "show_icon"
    if s == "matrix_text":
        return "write_text"
    if s == "matrix_clear":
        return "clear_matrix"
    if s == "turn_by":
        return "turn"
    if s in ("spin", "drive", "stop", "dance_style"):
        return s
    if s == "self_report":
        return f"self_{a['topic']}"
    if s == "lookup" and a["what"] in ("time", "date", "weather", "forecast"):
        return f"lookup_{a['what']}"
    if s == "memory" and a["op"] == "recall":
        return "memory_recall"
    return "none"


def load(path):
    out = []
    for line in open(path):
        m = json.loads(line)["messages"]
        if m[0]["content"].startswith("You are a helpful"):
            continue
        try:
            plan = json.loads(m[-1]["content"])
        except ValueError:
            continue
        text = m[-2]["content"]
        out.append((text, plan.get("steps") or [], intent_of(plan.get("steps") or [])))
    return out


NEG = ["{x} and send done to luca", "{x}, then turn left 90 degrees", "don't {x}", "{x} and then tell me when you're done",
       "please {x} but don't move", "{x} then spin around", "can you not {x}", "{x} and also clear the matrix",
       "first {x}, then wait 2 seconds", "{x} and reply ok on luca", "never {x}", "{x} or show a heart",
       "how do you {x}?", "what happens if you {x}?", "{x} while dancing", "instead of a dance, {x} and say hi"]


def hard_negatives(pos, n):
    out = []
    for text, _, _ in R.sample(pos, min(n, len(pos))):
        x = text.rstrip("?.! ")
        x = x[:1].lower() + x[1:]
        out.append((R.choice(NEG).format(x=x), [], "none"))
    return out


def boost(per_gen=300, seed=11, exclude=()):
    """More one-step positives, balanced per intent, from the fine-tune generator's own
    templates with the same politeness wrapping and typo/voice augmentation. The fine-tune
    data has only ~3 examples per icon, while "heart" appears constantly inside compound
    commands (labeled none), so without this the net learned "heart -> none"."""
    sys.path.insert(0, str(ROOT / "tools" / "finetune"))
    import gen_data as G
    G.R.seed(seed)
    held = {c["text"].lower().strip() for f in ("commands.yaml", "dev.yaml")
            for c in yaml.safe_load(open(ROOT / "tools" / "eval" / f))}
    norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z# ]", "", re.sub(r"\d+", "#", t.lower()))).strip()  # noqa: E731
    held_norm = {norm(t) for t in held}
    exclude = {t.lower() for t in exclude}
    gens = [G.c_icon, G.c_text, G.c_clear, G.c_turn, G.c_turn_idiom, G.c_spin, G.c_drive, G.c_stop,
            G.c_dance_style, G.c_self, G.c_lookup, G.c_memory]
    out, seen = [], set()
    for gen in gens:
        quota = per_gen * (4 if gen is G.c_icon else 1)     # 24 icons share one generator
        n = 0
        for _ in range(quota * 20):
            if n >= quota:
                break
            bare, steps = gen()
            # Both the bare command ("show a heart", the commonest real input, which the
            # generator always wraps in politeness) and the wrapped, augmented version.
            for text, st in ((bare, steps), G.augment(G.wrap(bare), steps)):
                k = text.lower().strip()
                if k in seen or k in held or norm(k) in held_norm or k in exclude:
                    continue
                seen.add(k)
                out.append((text, st, intent_of(st)))
                n += 1
    return out


def same_plan(a, b, text=""):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x["skill"] != y["skill"]:
            return False
        ax, ay = dict(x["args"]), dict(y["args"])
        if x["skill"] == "drive":
            cls = lambda s: (s > 0, 0 if abs(s) <= 30 else 2 if abs(s) >= 90 else 1)  # noqa: E731
            if cls(ax.pop("speed")) != cls(ay.pop("speed")):
                return False
        if x["skill"] == "memory":
            ax["key"], ay["key"] = (k.lower().replace("'", "") for k in (ax["key"], ay["key"]))
        if x["skill"] == "matrix_text":
            ax["text"], ay["text"] = ax["text"].lower(), ay["text"].lower()
        if x["skill"] == "dance_style":
            ax["style"], ay["style"] = ax["style"].lower(), ay["style"].lower()
            if not re.search(r"\d{2,3}\s*(bpm|beats)|\bbpm\b", text.lower()):
                ax.pop("bpm"), ay.pop("bpm")   # gen_data bug: some templates drop the tempo from the text
        if ax != ay:
            return False
    return True


def train(rows, epochs=10, batch=64, seed=0):
    net = Net(seed=seed)
    opt = Adam(net.params())
    X = [features(normalize(t)[0]) for t, _, _ in rows]
    y = np.array([INTENTS.index(i) for _, _, i in rows])
    idx = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(idx)
        losses = []
        for k in range(0, len(idx), batch):
            b = idx[k:k + batch]
            loss, g = net.loss_and_grads([X[i] for i in b], y[b])
            opt.step(g)
            losses.append(loss)
        acc = (net.forward(X)[0].argmax(1) == y).mean()
        print(f"epoch {ep + 1}: loss {np.mean(losses):.4f}, train accuracy {acc:.3f}", flush=True)
    return net


def evaluate(router, rows):
    """-> per-(tau, margin) accepted and correct counts, on (text, gold_steps, intent) rows."""
    decisions = []
    saved = (router.tau, router.margin)
    router.tau, router.margin = 0.0, 0.0
    for text, steps, intent in rows:
        t = normalize(text)[0]
        d = router.route(t)
        right = d["fast"] and same_plan(d["plan"]["steps"], steps, text)
        decisions.append((d["fast"], d["p"], d["margin"], right, intent != "none"))
    router.tau, router.margin = saved
    return decisions


def calibrate(decisions):
    best = None
    pos = sum(1 for d in decisions if d[4]) or 1
    for tau in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        for margin in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]:
            acc = [d for d in decisions if d[0] and d[1] >= tau and d[2] >= margin]
            if not acc:
                continue
            prec = sum(d[3] for d in acc) / len(acc)
            cov = len(acc) / pos
            wrong = len(acc) - sum(d[3] for d in acc)
            if prec >= PRECISION_TARGET and (best is None or cov > best["coverage"]):
                best = {"tau": tau, "margin": margin, "precision": round(prec, 4), "coverage": round(cov, 4),
                        "accepted": len(acc), "wrong": wrong}
    return best


def eval_set(router, name):
    from run_eval import score
    cases = yaml.safe_load(open(ROOT / "tools" / "eval" / f"{name}.yaml")) or []
    acc, right, ms, wrong = 0, 0, [], []
    for c in cases:
        t = normalize(c["text"])[0]
        d = router.route(t)
        ms.append(d["ms"])
        if d["fast"]:
            acc += 1
            ok = score(c, d["plan"])[0] == "correct"
            right += ok
            if not ok:
                wrong.append((c["id"], c["text"], d["plan"]["steps"]))
    return {"n": len(cases), "accepted": acc, "correct": right,
            "precision": round(right / acc, 4) if acc else None, "coverage": round(acc / max(1, len(cases)), 4),
            "route_ms_mean": round(float(np.mean(ms)), 3) if ms else None, "wrong": wrong}


def main():
    t0 = time.time()
    tr, va = load(DATA / "train.jsonl"), load(DATA / "val.jsonl")
    cal_dir = ROOT / "tools" / "fastpath" / "calib_data"
    if not (cal_dir / "train.jsonl").exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "tools" / "finetune" / "gen_data.py"), "--seed", "7",
                        "--n", "20000", "--val", "0", "--out", str(cal_dir)], check=True, capture_output=True)
    # The boost may use any phrasing, even the commonest ("show a heart"); calibration below
    # drops whatever training has seen, so its precision still measures unseen commands.
    tr = tr + boost(exclude=[t for t, _, _ in va])
    pos = [r for r in tr if r[2] != "none"]
    rows = tr + hard_negatives(pos, len(pos))
    counts = {i: sum(1 for r in rows if r[2] == i) for i in INTENTS}
    print(f"{len(rows)} training rows ({counts['none']} none), {len(va)} validation rows", flush=True)
    net = train(rows)
    OUT.mkdir(exist_ok=True)
    net.save(OUT / "model.npz")
    router = Router(OUT / "model.npz", calibration=OUT / "missing.json")
    # Calibration: a fresh 20,000-command set from another seed (minus anything in training),
    # so precision rests on thousands of accepted commands, not dozens.
    seen = {t.lower() for t, _, _ in tr}
    fresh = [r for r in load(cal_dir / "train.jsonl") if r[0].lower() not in seen]
    va_rows = va + fresh + hard_negatives([r for r in fresh if r[2] != "none"], 2000)
    print(f"calibrating on {len(va_rows)} commands ({sum(r[2] != 'none' for r in va_rows)} eligible)", flush=True)
    cal = calibrate(evaluate(router, va_rows))
    print("calibration on val:", cal, flush=True)
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1))
    router = Router(OUT / "model.npz", OUT / "calibration.json")
    report = {"date": time.strftime("%Y-%m-%d"), "params": net.n_params(), "train_rows": len(rows),
              "calibration": cal, "held_out": eval_set(router, "commands"), "dev": eval_set(router, "dev"),
              "train_s": round(time.time() - t0)}
    for k in ("held_out", "dev"):
        r = report[k]
        print(f"{k}: accepted {r['accepted']}/{r['n']} ({r['coverage']:.0%}), correct {r['correct']}, "
              f"precision {r['precision']}, route {r['route_ms_mean']} ms", flush=True)
        for w in r["wrong"]:
            print("   WRONG", w, flush=True)
    (ROOT / "docs" / "bench" / "fastpath.json").write_text(json.dumps(report, indent=1))
    print(f"{net.n_params()} parameters, model {(OUT / 'model.npz').stat().st_size / 1e3:.0f} kB, {report['train_s']} s")


if __name__ == "__main__":
    main()
