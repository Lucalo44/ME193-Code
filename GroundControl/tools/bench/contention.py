#!/usr/bin/env python3
"""CPU budget (Part 2, A0): how much a perception model slows the planner, with and without
core pinning. Until the real models exist (keyword spotting, pose, beat tracking), a numpy
matmul loop in the app container stands in for one, at a chosen duty cycle.

    adb forward tcp:9999 tcp:9999
    venv/bin/python tools/bench/contention.py              # writes docs/bench/<date>_contention.json

Conditions: no load; one load unpinned; one load pinned to core 3; two loads unpinned.
For each, the same commands are planned and decode tok/s and total time are recorded.
"""
import json
import subprocess
import time

from common import ROOT, load_cases, make_planner, plan_one

import config

CONTAINER = "ground-control-main-1"
CMDS = "s1,s5,m1,d1"
# Pin BEFORE importing numpy: BLAS starts its worker threads at import, and they keep the
# affinity they were born with.
LOAD = ("import os,time\n"
        "pin=os.environ.get('PIN')\n"
        "if pin: os.sched_setaffinity(0,{int(pin)})\n"
        "import numpy as np\n"
        "a=np.random.rand(160,160).astype(np.float32)\n"
        "duty=float(os.environ.get('DUTY','1'))\n"
        "while True:\n"
        "  t=time.time()\n"
        "  while time.time()-t<0.1*duty: a=a@a; a/=a.max()\n"
        "  time.sleep(0.1*(1-duty))\n")


def start_load(pin=None, duty=1.0, threads=None, tag="gcload"):
    env = f"-e DUTY={duty}" + (f" -e PIN={pin}" if pin is not None else "")
    if threads:
        env += f" -e OPENBLAS_NUM_THREADS={threads} -e OMP_NUM_THREADS={threads}"
    subprocess.run(["adb", "shell", f"docker exec -d {env} {CONTAINER} python3 -c \"{LOAD}\" {tag}"],
                   check=False, capture_output=True)


def stop_loads():
    subprocess.run(["adb", "shell", f"docker exec {CONTAINER} pkill -f 'numpy as np' || true"],
                   check=False, capture_output=True)


def measure(pl, rec, cases):
    rows = [plan_one(pl, rec, c)[0] for c in cases]
    tok_s = [r["decode_n"] / (r["decode_ms"] / 1000) for r in rows if r["decode_ms"]]
    pre = [r["prompt_n"] / (r["prefill_ms"] / 1000) for r in rows if r["prefill_ms"] and r["prompt_n"]]
    return {"decode_tok_s": round(sum(tok_s) / len(tok_s), 2), "prefill_tok_s": round(sum(pre) / len(pre), 2),
            "mean_ms": round(sum(r["total_ms"] for r in rows) / len(rows)),
            "correct": sum(r["outcome"] == "correct" for r in rows)}


def main():
    cases = load_cases(only=CMDS)
    pl, rec, _ = make_planner("http://127.0.0.1:9999", config.LLM_MODEL)
    pl.warm()
    import sys
    conds = [("no load", []),
             ("1 load, BLAS on all cores (default)", [dict()]),
             ("1 load, 1 thread, unpinned", [dict(threads=1)]),
             ("1 load, 1 thread, pinned to core 3", [dict(threads=1, pin=3)]),
             ("1 load at 30% duty, 1 thread, pinned to core 3", [dict(threads=1, pin=3, duty=0.3)])]
    if "--quick" in sys.argv:
        conds = [c for c in conds if "1 thread" in c[0] or c[0] == "no load"]
    out = []
    for name, loads in conds:
        stop_loads()
        for ld in loads:
            start_load(**ld)
        time.sleep(2)
        m = measure(pl, rec, cases)
        stop_loads()
        out.append({"condition": name, **m})
        print(f"{name:<38} decode {m['decode_tok_s']:.2f} tok/s  prefill {m['prefill_tok_s']:.2f} tok/s  "
              f"mean {m['mean_ms'] / 1000:.1f} s  correct {m['correct']}/{len(cases)}", flush=True)
    path = ROOT / "docs" / "bench" / f"{time.strftime('%Y-%m-%d')}_contention.json"
    path.write_text(json.dumps({"commands": CMDS, "results": out}, indent=1))
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
