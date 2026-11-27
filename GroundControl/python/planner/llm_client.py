"""Direct HTTP client for the board's llama.cpp server (OpenAI-compatible).

Stdlib only (urllib), so it runs in the app container and on a laptop alike.
Everything here was verified on the board's server build in Phase 0
(docs/SPIKES.md): streaming + json_schema + top_logprobs together, /tokenize
with pieces, /apply-template, and timings.cache_n.
"""
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

import config


class LlmError(RuntimeError):
    pass


class LlmClient:
    def __init__(self, url=None, model=None):
        self.url = (url or config.LLM_URL).rstrip("/")
        self.model = model or config.LLM_MODEL

    # ---- plumbing ----
    def _post(self, path, body, timeout=config.LLM_TIMEOUT_S, stream=False):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise LlmError(f"{path}: HTTP {e.code} {e.read()[:300]!r}") from None
        except urllib.error.URLError as e:
            raise LlmError(f"{path}: {e.reason}") from None
        return r if stream else json.loads(r.read())

    def models(self):
        r = urllib.request.urlopen(self.url + "/v1/models", timeout=10)
        return [m["id"] for m in json.loads(r.read())["data"]]

    def tokenize(self, text):
        d = self._post("/tokenize", {"model": self.model, "content": text, "with_pieces": True,
                                     "add_special": False, "parse_special": True}, timeout=30)
        return [{"id": t["id"], "piece": t["piece"] if isinstance(t["piece"], str) else ""}
                for t in d["tokens"]]

    def apply_template(self, messages):
        return self._post("/apply-template", {"model": self.model, "messages": messages}, timeout=30)["prompt"]

    # ---- saved prompt caches (Part 2, A1) ----
    def props(self):
        r = urllib.request.urlopen(self.url + "/props", timeout=10)
        return json.loads(r.read())

    def slots(self):
        r = urllib.request.urlopen(f"{self.url}/slots?model={urllib.parse.quote(self.model)}", timeout=10)
        return json.loads(r.read())

    def slot(self, action, filename, slot_id=0, timeout=120):
        """Save or restore slot 0's KV cache to/from a file in the server's --slot-save-path.
        -> the server's reply, e.g. {"n_saved": 874, "timings": {"save_ms": ...}}."""
        return self._post(f"/slots/{slot_id}?action={action}", {"model": self.model, "filename": filename},
                          timeout=timeout)

    # ---- generation ----
    def chat_stream(self, messages, schema=None, max_tokens=config.LLM_MAX_TOKENS,
                    on_token=None, temperature=0.0, top_k=config.LLM_TOP_K, cancel=None):
        """Stream a completion. Calls on_token(tok) per generated token where
        tok = {i, piece, logprob, p, top: [{piece, p}], t_ms}. Returns
        {text, tokens, timings, wall_ms, first_token_ms}."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "stream": True, "temperature": temperature, "cache_prompt": True,
                "timings_per_token": True}
        if top_k > 0:
            body["logprobs"] = True
            body["top_logprobs"] = top_k
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "plan", "schema": schema}}
        t0 = time.time()
        resp = self._post("/v1/chat/completions", body, stream=True)
        toks, text, timings, first = [], [], {}, None
        try:
            for raw in resp:
                if cancel is not None and cancel.is_set():
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                ch = json.loads(data)
                if ch.get("timings"):
                    timings = ch["timings"]
                for c in ch.get("choices") or []:
                    lp = (c.get("logprobs") or {}).get("content") or []
                    if lp:
                        for e in lp:
                            tok = self._tok(len(toks), e, t0)
                            first = first or tok["t_ms"]
                            toks.append(tok)
                            text.append(tok["piece"])
                            if on_token:
                                on_token(tok)
                    else:
                        piece = (c.get("delta") or {}).get("content")
                        if piece:
                            tok = {"i": len(toks), "piece": piece, "logprob": None, "p": None, "top": [],
                                   "t_ms": round((time.time() - t0) * 1000)}
                            first = first or tok["t_ms"]
                            toks.append(tok)
                            text.append(piece)
                            if on_token:
                                on_token(tok)
        finally:
            resp.close()
        return {"text": "".join(text), "tokens": toks, "timings": timings,
                "wall_ms": round((time.time() - t0) * 1000), "first_token_ms": first}

    @staticmethod
    def _tok(i, e, t0):
        top = [{"piece": x["token"], "p": round(math.exp(x["logprob"]), 4)}
               for x in e.get("top_logprobs") or []]
        lp = e.get("logprob")
        return {"i": i, "piece": e["token"], "logprob": lp,
                "p": round(math.exp(lp), 4) if lp is not None else None,
                "top": top, "t_ms": round((time.time() - t0) * 1000)}

    def chat_tools(self, messages, tools, max_tokens=200, temperature=0.0):
        """Non-streamed native tool calling (Baseline B, what the LLM Brick does)."""
        t0 = time.time()
        d = self._post("/v1/chat/completions", {
            "model": self.model, "messages": messages, "tools": tools, "max_tokens": max_tokens,
            "temperature": temperature, "cache_prompt": True})
        msg = d["choices"][0]["message"]
        return {"message": msg, "timings": d.get("timings", {}), "wall_ms": round((time.time() - t0) * 1000)}
