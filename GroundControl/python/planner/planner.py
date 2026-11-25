"""command -> Plan, in ONE constrained generation, with the visualization feed.

For the dashboard it emits:
  llm.prompt  the prompt as token chips by segment, and how many are KV-cached
  llm.token   each generated token, its top-5 alternatives, entropy, which JSON
              field it belongs to, and whether it's a "decision" token (the first
              token of a skill / mode / move / icon choice)
  llm.done    the plan, raw text, timings (prefill/decode/cache)
"""
import json
import math
import re
import threading
import time

import config
from dance.moves import CAR_MOVES, LEGO_MOVES, MODES
from planner import prompt as P
from planner.llm_client import LlmError
from skills.font import ICON_NAMES
from skills.registry import REGISTRY, plan_schema, tools_spec

DECISION_KEYS = {"skill", "mode", "car", "lego", "name", "topic"}


def choices_for(key):
    return {"skill": list(REGISTRY), "skills": list(REGISTRY), "mode": MODES, "car": CAR_MOVES, "lego": LEGO_MOVES,
            "name": ICON_NAMES}.get(key, [])


class JsonFieldTracker:
    """Incremental JSON scanner: which key's value is each generated char in?"""

    def __init__(self):
        self.stack = []          # '{' or '['
        self.in_str = False
        self.esc = False
        self.is_key = False
        self.key_buf = ""
        self.cur_key = None
        self.expect_key = False
        self.val_key = None      # key whose value we're inside (string/number)
        self.val_pos = 0         # chars into the current value

    def feed(self, piece):
        """-> (key, first_value_char_in_piece) for the piece."""
        label, starts = None, False
        for ch in piece:
            if self.in_str:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
                    if self.is_key:
                        self.cur_key = self.key_buf
                        self.is_key = False
                    else:
                        self.val_key = None
                    continue
                if self.is_key:
                    self.key_buf += ch
                elif self.val_key:
                    label = self.val_key
                    if self.val_pos == 0:
                        starts = True
                    self.val_pos += 1
                continue
            if ch == '"':
                self.in_str = True
                top = self.stack[-1] if self.stack else None
                if top == "{" and self.expect_key:
                    self.is_key, self.key_buf = True, ""
                    self.expect_key = False
                else:
                    self.val_key, self.val_pos = self.cur_key, 0
            elif ch == "{":
                self.stack.append("{")
                self.expect_key = True
            elif ch == "[":
                self.stack.append("[")
            elif ch in "}]":
                if self.stack:
                    self.stack.pop()
            elif ch == ",":
                if self.stack and self.stack[-1] == "{":
                    self.expect_key = True
            elif ch in "-0123456789." and self.cur_key:
                label = self.cur_key
        return label, starts


def entropy_bits(top):
    ps = [t["p"] for t in top if t.get("p")]
    rest = max(0.0, 1.0 - sum(ps))
    if rest > 1e-6:
        ps.append(rest)
    return round(-sum(p * math.log2(p) for p in ps if p > 0), 3)


def group_alternatives(key, top):
    """Map token alternatives onto the choices they could start, e.g.
    'matrix' -> matrix_text|matrix_icon|matrix_clear."""
    opts = choices_for(key)
    groups = {}
    for t in top:
        frag = t["piece"].lstrip(' "')
        if not frag:
            continue
        hits = [o for o in opts if o.startswith(frag)]
        label = "|".join(hits) if hits else f"({frag})"
        groups[label] = round(groups.get(label, 0) + t["p"], 4)
    return sorted(({"choice": k, "p": v} for k, v in groups.items()), key=lambda x: -x["p"])


def declared_in_think(think):
    """Skill names the model wrote in its think, in order of appearance."""
    hits = []
    for name in REGISTRY:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", think or ""):
            hits.append((m.start(), name))
    return [n for _, n in sorted(hits)]


class Planner:
    def __init__(self, client, emit=None, think_chars=None, style=None):
        self.client = client
        self.style = style or config.PLAN_STYLE
        # One generation at a time. The boot warm-up runs on its own thread; without
        # this, a command arriving mid-warm-up made llama.cpp run both side by side
        # in a shared context and prefill crawled at <3 tok/s.
        self.llm_lock = threading.Lock()
        self.emit = emit or (lambda *a, **k: None)
        self.think_chars = config.THINK_MAX_CHARS if think_chars is None else think_chars
        self._last_prompt_ids = []
        self.last_tok_s = None

    # ---- boot ----
    def warm(self):
        """Prefill the fixed prefix once so the first real command hits the cache."""
        with self.llm_lock:
            return self._warm()

    def _warm(self):
        t0 = time.time()
        msgs = P.messages("show a smile", self.think_chars, self.style)
        r = self.client.chat_stream(msgs, schema=plan_schema(self.think_chars, style=self.style),
                                    max_tokens=8, top_k=0)
        self.prompt_view("show a smile")       # remember the warm prompt's tokens for the cache view
        return {"ms": round((time.time() - t0) * 1000), "timings": r["timings"]}

    # ---- prompt chips ----
    def prompt_view(self, command):
        msgs = P.messages(command, self.think_chars, self.style)
        try:
            full = self.client.apply_template(msgs)
            toks = self.client.tokenize(full)
        except (LlmError, OSError, KeyError):
            return None
        spans = []
        for kind, text in P.segments(self.think_chars, self.style) + [("command", command)]:
            at = full.find(text)
            if at >= 0:
                spans.append((at, at + len(text), kind))
        pos, out = 0, []
        for t in toks:
            kind = "template"
            for a, b, k in spans:
                if a <= pos < b:
                    kind = k
                    break
            out.append({"id": t["id"], "piece": t["piece"], "kind": kind})
            pos += len(t["piece"])
        ids = [t["id"] for t in out]
        n_cached = 0
        for a, b in zip(ids, self._last_prompt_ids):
            if a != b:
                break
            n_cached += 1
        self._last_prompt_ids = ids
        return {"tokens": out, "n_tokens": len(out), "n_cached_predicted": n_cached}

    # ---- main path ----
    def plan(self, job, max_retries=1):
        with self.llm_lock:
            return self._plan(job, max_retries)

    def _plan(self, job, max_retries=1):
        msgs = P.messages(job.text, self.think_chars, self.style)
        schema = plan_schema(self.think_chars, config.MAX_STEPS, style=self.style)
        pv = self.prompt_view(job.text)
        if pv:
            self.emit("llm.prompt", {"id": job.id, **pv})
        last_err = None
        for attempt in range(max_retries + 1):
            tracker = JsonFieldTracker()

            def on_token(tok):
                key, starts = tracker.feed(tok["piece"])
                tok["key"] = key
                tok["decision"] = bool(starts and key in DECISION_KEYS)
                tok["entropy"] = entropy_bits(tok["top"]) if tok["top"] else None
                if tok["top"]:
                    raw_best = tok["top"][0]
                    tok["forced"] = raw_best["piece"] != tok["piece"] and raw_best["p"] > (tok["p"] or 0)
                if tok["decision"] and tok["top"]:
                    tok["choices"] = group_alternatives(key, tok["top"])
                self.emit("llm.token", {"id": job.id, "attempt": attempt, **tok})

            t0 = time.time()
            r = self.client.chat_stream(msgs, schema=schema, on_token=on_token, cancel=job.cancel)
            job.mark("llm", t0)
            t = r["timings"] or {}
            timings = {"prefill_ms": round(t.get("prompt_ms", 0)), "prompt_n": t.get("prompt_n"),
                       "cache_n": t.get("cache_n"), "decode_ms": round(t.get("predicted_ms", 0)),
                       "predicted_n": t.get("predicted_n"),
                       "tok_s": round(t.get("predicted_per_second", 0) or 0, 2),
                       "prefill_tok_s": round(t.get("prompt_per_second", 0) or 0, 1),
                       "wall_ms": r["wall_ms"], "first_token_ms": r["first_token_ms"]}
            self.last_tok_s = timings["tok_s"] or self.last_tok_s
            try:
                plan = json.loads(r["text"])
                if not isinstance(plan.get("steps"), list):
                    raise ValueError("no steps")
                actual = [st.get("skill") for st in plan["steps"]]
                if self.style == "parts":
                    parts = plan.get("parts") or []
                    need = bool(parts) and len(parts) != len(actual)
                    fixed = parts
                else:
                    declared = declared_in_think(plan.get("think", ""))
                    need = bool(declared) and declared != actual
                    fixed = (plan.get("think", ""), declared)
                if need and not job.cancel.is_set():
                    plan, r2 = self._repair(job, msgs, plan, fixed, on_token)
                    if r2:
                        t2 = r2["timings"] or {}
                        timings["repair_ms"] = r2["wall_ms"]
                        timings["decode_ms"] += round(t2.get("predicted_ms", 0))
                        timings["prefill_ms"] += round(t2.get("prompt_ms", 0))
                        timings["predicted_n"] = (timings["predicted_n"] or 0) + (t2.get("predicted_n") or 0)
                        timings["wall_ms"] += r2["wall_ms"]
                        r["tokens"] = r["tokens"] + r2["tokens"]
                        r["text"] = r2["text"]
                decisions = [{"i": x["i"], "key": x["key"], "piece": x["piece"], "choices": x.get("choices")}
                             for x in r["tokens"] if x.get("decision")]
                self.emit("llm.done", {"id": job.id, "plan": plan, "raw_text": r["text"],
                                       "timings": timings, "attempt": attempt, "decisions": decisions})
                return plan, {"timings": timings, "raw": r["text"], "attempts": attempt + 1,
                              "n_tokens": len(r["tokens"])}
            except (ValueError, json.JSONDecodeError) as e:
                last_err = f"unparseable plan ({e}): {r['text'][:120]!r}"
                self.emit("llm.retry", {"id": job.id, "attempt": attempt, "error": last_err})
                if job.cancel.is_set():
                    break
        self.emit("llm.done", {"id": job.id, "plan": None, "error": last_err})
        return None, {"error": last_err}

    def _repair(self, job, msgs, plan, fixed, on_token):
        """The steps don't match what the model declared first (its parts, or the
        skills named in its think). Re-run once with that declaration pinned and
        the step slots forced to match it."""
        n = len(fixed) if self.style == "parts" else len(fixed[1])
        self.emit("llm.repair", {"id": job.id, "style": self.style,
                                 "declared": fixed if self.style == "parts" else fixed[1],
                                 "got": [st.get("skill") for st in plan["steps"]]})
        schema = plan_schema(self.think_chars, config.MAX_STEPS, fixed=fixed, style=self.style)
        r2 = self.client.chat_stream(msgs, schema=schema, on_token=on_token, cancel=job.cancel)
        try:
            p2 = json.loads(r2["text"])
            if len(p2.get("steps", [])) == n:
                p2["repaired"] = True
                return p2, r2
        except (ValueError, json.JSONDecodeError):
            pass
        return plan, None

    # ---- answer composer: a direct reply to the exact question, from gathered facts
    COMPOSE_SYSTEM = (
        "You are Ground Control, a small robot built on an Arduino UNO Q. Answer the user's question "
        "directly in one or two natural sentences, speaking as yourself (I, my). Start with exactly what was "
        "asked: a number, a name or a value. For a yes/no question, answer yes or no exactly as the facts say. Then add the "
        "one most useful detail from the facts. Keep it under 30 words. Use only the facts given; "
        "never invent numbers, names or devices. If the facts don't answer the question, say so briefly and "
        "give the closest fact.")

    def compose(self, question, facts, job_id=None, cancel=None, max_tokens=90):
        """One short free-text generation. Returns the answer, or None on failure
        (callers fall back to their deterministic text)."""
        msgs = [{"role": "system", "content": self.COMPOSE_SYSTEM},
                {"role": "user", "content": f"Facts: {facts}\nQuestion: {question}"}]
        with self.llm_lock:
            self.emit("llm.compose", {"id": job_id, "facts": facts[:600]})

            def on_token(tok):
                self.emit("llm.token", {"id": job_id, "attempt": 0, "compose": True, **tok})
            try:
                r = self.client.chat_stream(msgs, schema=None, max_tokens=max_tokens, temperature=0.1,
                                            on_token=on_token, cancel=cancel)
            except Exception as e:  # noqa: BLE001
                print(f"[compose] failed: {e!r}", flush=True)
                return None
        text = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S).strip().strip('"')
        text = re.sub(r"\s+", " ", text)
        # This model opens almost everything with "Yes." Keep it only for yes/no questions.
        yes_no = re.match(r"^\s*(is|are|am|do|does|did|can|could|will|would|should|shall|has|have|had|was|were)\b",
                          question, re.I)
        if not yes_no:
            text = re.sub(r"^(yes|no)[.,!]?\s+", "", text, flags=re.I)
            text = text[:1].upper() + text[1:]
        else:
            # "Yes. Nothing is plugged in." contradicts itself; the sentence after it is what the facts say.
            text = re.sub(r"^yes[.,!]?\s+(?=(nothing|no |none|not |there (is|are) no|i (do not|don't|have no)))",
                          "No. ", text, flags=re.I)
        return text or None

    # ---- Baseline B: native tool calling loop (what the LLM Brick does) ----
    def plan_with_tools(self, text, max_rounds=4):
        msgs = [{"role": "system", "content": P.HEAD.split("\n")[0] + " Use the tools to do what the user asks."},
                {"role": "user", "content": text}]
        steps, wall, t0 = [], 0, time.time()
        for _ in range(max_rounds):
            r = self.client.chat_tools(msgs, tools_spec())
            wall += r["wall_ms"]
            calls = r["message"].get("tool_calls") or []
            if not calls:
                break
            msgs.append(r["message"])
            for c in calls:
                fn = c.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                steps.append({"skill": fn.get("name"), "args": args})
                result = {"yaw": 90.0, "pitch": 0.0, "roll": 0.0} if fn.get("name") == "read_imu" else "ok"
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(result)})
        return {"steps": steps}, {"wall_ms": round((time.time() - t0) * 1000)}
