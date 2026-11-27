"""Shared instrumentation for the latency benchmark (tools/bench/latency.py) and the
trace recorder (tools/trace/record.py): plan one command against the real model and
capture everything the planner emits, without changing the planner.

Needs `adb forward tcp:9999 tcp:9999` (and tcp:8770 for the app-name grounding, so the
prompt matches the board's).
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import config  # noqa: E402
import skills  # noqa: E402,F401
import yaml  # noqa: E402
from jobs import Job  # noqa: E402
from planner.llm_client import LlmClient  # noqa: E402
from planner.planner import Planner  # noqa: E402
from run_eval import score  # noqa: E402

MODELS_DIR = "/var/lib/arduino-app-cli/models/llamacpp"


def keep_awake():
    """macOS: idle sleep suspends the USB link to the board mid-run (adb forwards die, and
    commands come back "connection refused" or slow). Hold the Mac awake while we run."""
    import os
    import subprocess
    import sys
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        except OSError:
            pass


class Recorder:
    """Planner emit hook: keeps every event of the current command."""

    def __init__(self):
        self.events = []

    def __call__(self, kind, payload=None, **_):
        self.events.append((kind, payload or {}))

    def take(self):
        ev, self.events = self.events, []
        return ev


def load_cases(set_name="commands", only=None, limit=None):
    cases = yaml.safe_load(open(ROOT / "tools" / "eval" / f"{set_name}.yaml"))
    if isinstance(cases, dict):
        cases = cases.get("cases") or cases.get("commands") or []
    if only:
        keep = set(only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    return cases[:limit] if limit else cases


def make_planner(url, model, style=None):
    keep_awake()
    rec = Recorder()
    client = LlmClient(url, model)
    pl = Planner(client, emit=rec, think_chars=60, style=style or config.style_for(model))
    return pl, rec, client


def app_names():
    """Same app-name grounding as the board (needs adb forward tcp:8770)."""
    from lookup import Lookup, describe_apps_in_catalog
    try:
        return describe_apps_in_catalog(Lookup(host="127.0.0.1", port=8770), patch=False)
    except Exception:  # noqa: BLE001
        return None


def plan_one(pl, rec, case, names=None):
    """-> one benchmark row with timings, token split and outcome, plus the raw events."""
    from lookup import ground_app_questions
    rec.take()
    t0 = time.time()
    err = None
    from planner.normalize import normalize, turn_default_left
    try:
        text, _ = normalize(case["text"])                      # same input path as the board
        plan, info = pl.plan(Job(text, source="bench"))
        grounded_plan, note = ground_app_questions(plan, text, names) if plan else (plan, None)
        grounded_plan = turn_default_left(grounded_plan, text)
    except Exception as e:  # noqa: BLE001  record and keep going
        plan, info, grounded_plan, note, err = None, {}, None, None, repr(e)[:200]
    total_ms = round((time.time() - t0) * 1000)
    events = rec.take()
    toks = [p for k, p in events if k == "llm.token" and not p.get("compose")]
    think_n = sum(1 for t in toks if t.get("key") == "think")
    tm = info.get("timings") or {}
    outcome, why = score(case, grounded_plan) if not err else ("did_nothing", err)
    row = {
        "id": case["id"], "cat": case.get("cat"), "text": case["text"],
        "queue_ms": 0,                                     # the bench sends one command at a time
        "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
        "prefill_ms": tm.get("prefill_ms", 0), "decode_ms": tm.get("decode_ms", 0),
        "decode_n": tm.get("predicted_n") or len(toks),
        "think_tokens": think_n, "step_tokens": len(toks) - think_n,
        "repairs": sum(1 for k, _ in events if k == "llm.repair"),
        "retries": sum(1 for k, _ in events if k == "llm.retry"),
        "guard_hits": (["app_grounding"] if note else []),
        "total_ms": total_ms, "outcome": outcome, "why": why, "plan": grounded_plan,
    }
    return row, events, info


def gguf_hash(model, cache=ROOT / "docs" / "bench" / "gguf_hashes.json"):
    """sha256 of the model's GGUF on the board, cached by model id (hashing 1 GB takes ~15 s)."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    known = json.loads(cache.read_text()) if cache.exists() else {}
    if model in known:
        return known[model]
    path = f"{MODELS_DIR}/{model}.gguf"
    try:
        out = subprocess.run(["adb", "shell", f"sha256sum {path}"], capture_output=True, text=True, timeout=120)
        h = out.stdout.split()[0] if out.returncode == 0 and out.stdout else None
    except (OSError, subprocess.TimeoutExpired):
        h = None
    if h:
        known[model] = h
        cache.write_text(json.dumps(known, indent=1))
    return h


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"mean": None, "p50": None, "p90": None}
    pick = lambda q: xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]   # noqa: E731
    return {"mean": round(sum(xs) / len(xs)), "p50": pick(0.5), "p90": pick(0.9)}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]
