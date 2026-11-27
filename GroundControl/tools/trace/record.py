#!/usr/bin/env python3
"""Trace recorder (Part 2, A0): one full, reproducible trace per command, for the replay
view (B1) and the "why" view (B2).

    adb forward tcp:9999 tcp:9999; adb forward tcp:8770 tcp:8770
    venv/bin/python tools/trace/record.py --only m1,x1,cp1        # a few eval commands
    venv/bin/python tools/trace/record.py --model custom/gc-planner-v2-Q4_0 --all

Each trace (traces/<model>/<command_id>.json) holds the model id and GGUF hash, the prompt
as tokens with pieces, every generated token with its top-5 and logprob, the decision-token
indices, timings, the final plan, and whether it was right. Decoding is greedy
(temperature 0, the planner's default), so a rerun gives the same trace.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from common import ROOT, app_names, gguf_hash, load_cases, make_planner, plan_one  # noqa: E402

import config  # noqa: E402
from planner import prompt as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9999")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--set", default="commands")
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--where", default="board", help="where the model ran (board, or laptop llama.cpp)")
    a = ap.parse_args()
    if not a.only and not a.all:
        sys.exit("pick commands with --only id,id or --all")

    cases = load_cases(a.set, only=a.only or None)
    pl, rec, client = make_planner(a.url, a.model)
    names = app_names()
    pl.warm()
    h = gguf_hash(a.model) if a.where == "board" else None
    out_dir = ROOT / "traces" / a.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        msgs = P.messages(c["text"], pl.think_chars, pl.style)
        try:
            prompt_tokens = client.tokenize(client.apply_template(msgs))
        except Exception:  # noqa: BLE001
            prompt_tokens = None
        row, events, info = plan_one(pl, rec, c, names)
        toks = [p for k, p in events if k == "llm.token" and not p.get("compose")]
        trace = {
            "model": a.model, "gguf_sha256": h, "where": a.where, "style": pl.style,
            "recorded": time.strftime("%Y-%m-%d %H:%M"), "temperature": 0,
            "command": {"id": c["id"], "cat": c.get("cat"), "text": c["text"], "expect": c.get("expect")},
            "prompt_tokens": prompt_tokens,
            "tokens": [{k: t.get(k) for k in ("i", "attempt", "piece", "logprob", "p", "top", "key", "decision",
                                                "forced", "entropy", "t_ms")} for t in toks],
            "decisions": [i for i, t in enumerate(toks) if t.get("decision")],
            "repaired": row["repairs"] > 0,
            "timings": info.get("timings"),
            "plan": row["plan"], "outcome": row["outcome"], "why": row["why"],
        }
        (out_dir / f"{c['id']}.json").write_text(json.dumps(trace, indent=1))
        print(f"{row['outcome']:<11} {c['id']:<5} {len(toks):>3} tokens  {c['text'][:60]}", flush=True)
    print(f"traces in {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
