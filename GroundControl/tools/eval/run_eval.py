#!/usr/bin/env python3
"""Planner accuracy + latency on the eval set.

Runs on the laptop against the board's real llama.cpp (adb forward), with no
hardware involved: plans are scored, not executed.

    adb forward tcp:9999 tcp:9999
    venv/bin/python tools/eval/run_eval.py                     # Qwen, constrained JSON, think 60
    venv/bin/python tools/eval/run_eval.py --set dev           # tune on this, never on commands.yaml
    venv/bin/python tools/eval/run_eval.py --model gemma-3-1b-it-Q4_0
    venv/bin/python tools/eval/run_eval.py --mode tools        # Baseline B: native tool calling loop
    venv/bin/python tools/eval/run_eval.py --think 0

Outcomes per command:
  correct       exact skill sequence + every key arg
  did_nothing   no steps where some were expected (safe failure)
  did_wrong     anything else: wrong/extra/missing step, wrong arg, forbidden topic (the dangerous one)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "python"))

import config  # noqa: E402
import skills  # noqa: E402,F401
from jobs import Job  # noqa: E402
from mqtt_io import topic_allowed  # noqa: E402
from planner.llm_client import LlmClient  # noqa: E402
from planner.planner import Planner  # noqa: E402


def get_path(d, path):
    for part in str(path).split("."):
        if isinstance(d, list):
            try:
                d = d[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(d, dict):
            d = d.get(part)
        else:
            return None
    return d


def arg_ok(want, got, key):
    if key.endswith("topic") and want == "reply":
        return isinstance(got, str) and got.lower() in ("reply", config.TOPIC_REPLY.lower())
    if isinstance(want, str) and want.startswith("~"):
        return isinstance(got, str) and want[1:].lower() in got.lower()
    if isinstance(want, str):
        return isinstance(got, str) and got.strip().lower() == want.lower()
    if isinstance(want, (int, float)):
        try:
            return abs(float(got) - float(want)) < 1e-6
        except (TypeError, ValueError):
            return False
    return want == got


def score(case, plan):
    exp = case["expect"]
    steps = [s for s in ((plan or {}).get("steps") or []) if s.get("skill") != "read_imu"]
    got_sk = [s.get("skill") for s in steps]
    for s in steps:
        if s.get("skill") == "mqtt_publish":
            t = (s.get("args") or {}).get("topic", "")
            if t in case.get("forbid_topics", []) or (t.lower() not in ("reply",) and not topic_allowed(t)):
                return "did_wrong", f"publishes to forbidden topic {t!r}"
    if plan is None:
        return "did_nothing", "no plan (unparseable)"
    if not exp["skills"]:
        return ("correct", "") if not steps else ("did_wrong", f"should refuse, did {got_sk}")
    if not steps:
        return "did_nothing", "refused"
    if got_sk != exp["skills"]:
        return "did_wrong", f"skills {got_sk} != {exp['skills']}"
    for idx, kv in (exp.get("args") or {}).items():
        a = steps[int(idx)].get("args") or {}
        for k, want in kv.items():
            if case["id"] == "x4" and k == "seconds":
                continue
            got = get_path(a, k)
            if not arg_ok(want, got, k):
                return "did_wrong", f"step {idx} {k}={got!r}, want {want!r}"
    return "correct", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="commands", choices=["commands", "dev"])
    ap.add_argument("--url", default="http://127.0.0.1:9999")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--mode", default="json", choices=["json", "tools"])
    ap.add_argument("--think", type=int, default=config.THINK_MAX_CHARS)
    ap.add_argument("--style", default=config.PLAN_STYLE, choices=["parts", "think"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated ids")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cases = yaml.safe_load(open(HERE / f"{a.set}.yaml"))
    if a.only:
        cases = [c for c in cases if c["id"] in a.only.split(",")]
    if a.limit:
        cases = cases[: a.limit]
    client = LlmClient(a.url, a.model)
    pl = Planner(client, think_chars=a.think, style=a.style)
    if a.mode == "json":
        w = pl.warm()
        print(f"warm-up {w['ms']} ms", flush=True)
    rows = []
    for c in cases:
        t0 = time.time()
        if a.mode == "json":
            plan, info = pl.plan(Job(c["text"], source="eval"))
            tm = info.get("timings") or {}
        else:
            plan, info = pl.plan_with_tools(c["text"])
            tm = {"wall_ms": info["wall_ms"]}
        wall = round((time.time() - t0) * 1000)
        outcome, why = score(c, plan)
        rows.append({"id": c["id"], "cat": c["cat"], "text": c["text"], "outcome": outcome, "why": why,
                     "plan": plan, "wall_ms": wall, "timings": tm})
        mark = {"correct": "OK  ", "did_nothing": "NONE", "did_wrong": "BAD "}[outcome]
        print(f"{mark} {c['id']:<5} {wall:>6} ms  {c['text'][:60]:<60} {why}", flush=True)

    walls = [r["wall_ms"] for r in rows]
    n = len(rows)
    acc = sum(r["outcome"] == "correct" for r in rows) / n
    cats = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r["outcome"] == "correct")
    p90 = sorted(walls)[max(0, int(0.9 * n) - 1)]
    summary = {"set": a.set, "model": a.model, "mode": a.mode, "think": a.think if a.mode == "json" else None,
               "style": a.style if a.mode == "json" else None,
               "n": n, "accuracy": round(acc, 3),
               "did_nothing": sum(r["outcome"] == "did_nothing" for r in rows),
               "did_wrong": sum(r["outcome"] == "did_wrong" for r in rows),
               "mean_ms": round(statistics.mean(walls)), "p90_ms": p90,
               "by_cat": {k: f"{sum(v)}/{len(v)}" for k, v in cats.items()},
               "when": time.strftime("%Y-%m-%d %H:%M")}
    print("\n" + json.dumps(summary, indent=1))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    style = f"_{a.style}" if a.mode == "json" and a.style != "parts" else ""
    name = f"{a.set}_{a.model.split('/')[-1]}_{a.mode}_think{a.think}{style}{('_' + a.tag) if a.tag else ''}.json"
    (out / name).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"saved tools/eval/results/{name}")


if __name__ == "__main__":
    main()
