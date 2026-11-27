#!/usr/bin/env python3
"""Latency benchmark (Part 2, A0): where the time goes on every command.

    adb forward tcp:9999 tcp:9999; adb forward tcp:8770 tcp:8770
    venv/bin/python tools/bench/latency.py                    # fixed 20-command subset
    venv/bin/python tools/bench/latency.py --all              # the whole held-out set
    venv/bin/python tools/bench/latency.py --model custom/gc-planner-v2-Q4_0 --tag before-a1

Per command: queue wait, prefill tokens and ms, decode tokens and ms, think vs step
tokens, repair passes, guard hits, total. Writes docs/bench/<date>[_tag].json and a
stacked bar chart next to it (.svg). Every later phase quotes these numbers.
"""
import argparse
import json
import time

from common import ROOT, app_names, gguf_hash, load_cases, make_planner, plan_one, stats

import config

# A fixed, mixed subset so runs are comparable: singles, chains, info, dances, tools, traps.
SUBSET = ("s1,s2,s4,s5,m1,m3,m4,i1,t1,d1,d2,ds1,q1,sk1,lk1,cp1,mm1,wk1,x1,n1")


def svg_chart(rows, path):
    """Stacked bars per command: prefill, think decode, step decode, other (repair, client)."""
    W, bar, gap, top = 900, 14, 4, 30
    H = top + len(rows) * (bar + gap) + 40
    tmax = max(r["total_ms"] for r in rows) or 1
    sx = (W - 260) / tmax
    colors = {"prefill": "#4c78a8", "think": "#f58518", "steps": "#54a24b", "other": "#bab0ac"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    x0 = 240
    for k, (name, c) in enumerate(colors.items()):
        out.append(f'<rect x="{x0 + k * 110}" y="8" width="10" height="10" fill="{c}"/>'
                   f'<text x="{x0 + k * 110 + 14}" y="17">{name}</text>')
    for i, r in enumerate(rows):
        y = top + i * (bar + gap)
        dec = r["decode_ms"] or 0
        n = max(1, r["think_tokens"] + r["step_tokens"])
        parts = {"prefill": r["prefill_ms"] or 0, "think": dec * r["think_tokens"] / n,
                 "steps": dec * r["step_tokens"] / n}
        parts["other"] = max(0, r["total_ms"] - sum(parts.values()))
        label = (r["id"] + " " + r["text"])[:38].replace("&", "&amp;").replace("<", "&lt;")
        out.append(f'<text x="4" y="{y + bar - 3}">{label}</text>')
        x = x0
        for name, ms in parts.items():
            w = ms * sx
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar}" fill="{colors[name]}"/>')
            x += w
        mark = "" if r["outcome"] == "correct" else " ✗"
        out.append(f'<text x="{x + 4:.1f}" y="{y + bar - 3}">{r["total_ms"] / 1000:.1f} s{mark}</text>')
    out.append(f'<text x="{x0}" y="{H - 12}">seconds per command (prefill = reading the prompt; '
               f'think/steps = writing the plan)</text></svg>')
    path.write_text("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9999")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--all", action="store_true", help="the whole held-out set, not the 20-command subset")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cases = load_cases(only=a.only or (None if a.all else SUBSET))
    pl, rec, client = make_planner(a.url, a.model)
    names = app_names()
    w = pl.warm()
    print(f"warm-up {w['ms']} ms (prompt_n {w['timings'].get('prompt_n')}, cache_n {w['timings'].get('cache_n')})",
          flush=True)
    rows = []
    for c in cases:
        row, _, _ = plan_one(pl, rec, c, names)
        rows.append(row)
        print(f"{'OK ' if row['outcome'] == 'correct' else 'BAD'} {row['id']:<5} {row['total_ms']:>6} ms  "
              f"prefill {row['prefill_ms']:>5} ms ({row['prompt_n']} tok, {row['cache_n']} cached)  "
              f"decode {row['decode_ms']:>6} ms ({row['think_tokens']} think + {row['step_tokens']} steps)"
              f"{'  repair' if row['repairs'] else ''}", flush=True)

    summary = {k: stats([r[k] for r in rows]) for k in
               ("total_ms", "prefill_ms", "decode_ms", "decode_n", "think_tokens", "step_tokens", "prompt_n")}
    tok_s = [r["decode_n"] / (r["decode_ms"] / 1000) for r in rows if r["decode_ms"]]
    out = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "model": a.model, "gguf_sha256": gguf_hash(a.model),
        "style": pl.style, "n": len(rows), "warm_ms": w["ms"],
        "accuracy": sum(r["outcome"] == "correct" for r in rows) / max(1, len(rows)),
        "repair_rate": sum(r["repairs"] > 0 for r in rows) / max(1, len(rows)),
        "decode_tok_s_mean": round(sum(tok_s) / len(tok_s), 2) if tok_s else None,
        "summary": summary, "rows": rows,
    }
    bench = ROOT / "docs" / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y-%m-%d") + (f"_{a.tag}" if a.tag else "")
    (bench / f"{stem}.json").write_text(json.dumps(out, indent=1))
    svg_chart(rows, bench / f"{stem}.svg")
    s = summary
    print(f"\n{len(rows)} commands, accuracy {out['accuracy']:.0%}, repairs {out['repair_rate']:.0%}")
    print(f"total    mean {s['total_ms']['mean']} ms  p50 {s['total_ms']['p50']}  p90 {s['total_ms']['p90']}")
    print(f"prefill  mean {s['prefill_ms']['mean']} ms   decode mean {s['decode_ms']['mean']} ms "
          f"({s['decode_n']['mean']} tokens: {s['think_tokens']['mean']} think + {s['step_tokens']['mean']} steps, "
          f"{out['decode_tok_s_mean']} tok/s)")
    print(f"saved docs/bench/{stem}.json and .svg")


if __name__ == "__main__":
    main()
