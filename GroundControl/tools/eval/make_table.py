#!/usr/bin/env python3
"""Print the markdown comparison table from tools/eval/results/*.json.

    venv/bin/python tools/eval/make_table.py [--tag final] [--set commands]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = ["single", "multi", "info", "topic", "dance", "question", "trap", "nonsense", "injection"]

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="final")
ap.add_argument("--set", default="commands")
a = ap.parse_args()

rows = []
for f in sorted((HERE / "results").glob(f"{a.set}_*_{a.tag}.json")):
    d = json.loads(f.read_text())
    s = d["summary"]
    label = f"{s['model']} · {'native tools (Baseline B)' if s['mode'] == 'tools' else 'constrained JSON'}"
    if s["mode"] == "json":
        st = s.get("style") or "parts"
        label += f" · {st}" if s["think"] else " · no parts/think"
    rows.append((label, s, d["rows"]))

print("| Config | Accuracy | Did nothing | Did wrong | Mean | p90 | " + " | ".join(CATS) + " |")
print("|---|---|---|---|---|---|" + "---|" * len(CATS))
for label, s, _ in rows:
    cats = " | ".join(s["by_cat"].get(c, "–") for c in CATS)
    print(f"| {label} | **{s['accuracy'] * 100:.0f}%** ({round(s['accuracy'] * s['n'])}/{s['n']}) | "
          f"{s['did_nothing']} | {s['did_wrong']} | {s['mean_ms'] / 1000:.1f} s | {s['p90_ms'] / 1000:.1f} s | {cats} |")

for label, s, rs in rows:
    print(f"\n<details><summary>{label}: failures</summary>\n")
    for r in rs:
        if r["outcome"] != "correct":
            print(f"- `{r['id']}` {r['outcome']}: \"{r['text']}\" → {r['why']}")
    print("\n</details>")
