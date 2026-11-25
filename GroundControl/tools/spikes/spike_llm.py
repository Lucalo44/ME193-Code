#!/usr/bin/env python3
"""Phase 0 spike: which llama.cpp features work on the board's server build?

Stdlib only, so it runs on the board host (python3 spike_llm.py) or in the app
container. Prints PASS/FAIL per feature plus the numbers we design around.

    python3 spike_llm.py [--url http://127.0.0.1:9999] [--model Qwen3.5-0.8B-Q4_0]
"""
import argparse
import json
import time
import urllib.request

P = argparse.ArgumentParser()
P.add_argument("--url", default="http://127.0.0.1:9999")
P.add_argument("--model", default="Qwen3.5-0.8B-Q4_0")
A = P.parse_args()

SCHEMA = {
    "type": "object",
    "properties": {
        "think": {"type": "string", "maxLength": 60},
        "steps": {"type": "array", "maxItems": 3, "items": {
            "type": "object",
            "properties": {"skill": {"enum": ["matrix_text", "mqtt_publish"]},
                           "args": {"type": "object"}},
            "required": ["skill", "args"]}},
    },
    "required": ["think", "steps"],
}
SYSTEM = ("You plan actions for a robot. Reply with ONE JSON object: "
          '{"think": short, "steps": [{"skill": ..., "args": {...}}]}. '
          "Skills: matrix_text(text), mqtt_publish(topic, message).")
CMD = "Type Hi on the LED matrix and then send back on topic luca the message 'I am done'"

results = []


def post(path, body, stream=False, timeout=300):
    req = urllib.request.Request(A.url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.loads(r.read())
    return r


def sse(resp):
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data: "):
            d = line[6:]
            if d == "[DONE]":
                return
            yield json.loads(d)


def check(name, fn):
    t0 = time.time()
    try:
        info = fn()
        results.append((name, True, info))
        print(f"PASS  {name:<38} {time.time() - t0:6.1f}s  {info}", flush=True)
    except Exception as e:  # noqa: BLE001
        results.append((name, False, repr(e)[:200]))
        print(f"FAIL  {name:<38} {time.time() - t0:6.1f}s  {repr(e)[:200]}", flush=True)


def msgs():
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CMD}]


def t_models():
    req = urllib.request.urlopen(A.url + "/v1/models", timeout=30)
    d = json.loads(req.read())
    return [m["id"] for m in d["data"]]


def t_plain():
    d = post("/v1/chat/completions", {"model": A.model, "messages": msgs(), "max_tokens": 40})
    t = d.get("timings", {})
    return {"text": d["choices"][0]["message"]["content"][:60], "timings": t}


def t_cache():
    d = post("/v1/chat/completions", {"model": A.model, "messages": msgs(), "max_tokens": 5,
                                      "cache_prompt": True})
    t = d.get("timings", {})
    assert "cache_n" in t, f"no cache_n in timings: {t}"
    return {"cache_n": t["cache_n"], "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms")}


def t_schema():
    d = post("/v1/chat/completions", {
        "model": A.model, "messages": msgs(), "max_tokens": 120,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "plan", "schema": SCHEMA}}})
    txt = d["choices"][0]["message"]["content"]
    plan = json.loads(txt)
    t = d.get("timings", {})
    return {"plan": plan, "tok_s": t.get("predicted_per_second")}


def t_logprobs_chat():
    d = post("/v1/chat/completions", {"model": A.model, "messages": msgs(), "max_tokens": 5,
                                      "logprobs": True, "top_logprobs": 5})
    lp = d["choices"][0]["logprobs"]["content"]
    first = lp[0]
    return {"tok": first["token"], "top": [(x["token"], round(x["logprob"], 2))
                                          for x in first["top_logprobs"]]}


def t_stream():
    n, first_ms, t0 = 0, None, time.time()
    for ch in sse(post("/v1/chat/completions", {"model": A.model, "messages": msgs(),
                                                 "max_tokens": 15, "stream": True}, stream=True)):
        if ch.get("choices") and ch["choices"][0].get("delta", {}).get("content"):
            n += 1
            first_ms = first_ms or (time.time() - t0) * 1000
    assert n > 0
    return {"chunks": n, "first_token_ms": round(first_ms)}


def t_stream_schema_logprobs():
    toks, last = [], None
    body = {"model": A.model, "messages": msgs(), "max_tokens": 120, "stream": True,
            "logprobs": True, "top_logprobs": 5,
            "timings_per_token": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "plan", "schema": SCHEMA}}}
    for ch in sse(post("/v1/chat/completions", body, stream=True)):
        last = ch
        c = ch.get("choices") or []
        if c and c[0].get("logprobs") and c[0]["logprobs"].get("content"):
            for e in c[0]["logprobs"]["content"]:
                toks.append((e["token"], len(e.get("top_logprobs", []))))
    assert toks, "no logprobs in stream"
    text = "".join(t for t, _ in toks)
    json.loads(text)
    return {"n_tokens": len(toks), "top_k_each": toks[0][1],
            "final_has_timings": bool(last and last.get("timings"))}


def t_completion_nprobs():
    d = post("/completion", {"model": A.model, "prompt": "The robot said", "n_predict": 4,
                             "n_probs": 5, "post_sampling_probs": True})
    p = d["completion_probabilities"][0]
    keys = list(p.keys())
    return {"keys": keys, "top": [(x.get("token"), round(x.get("prob", x.get("logprob", 0)), 3))
                                  for x in p.get("top_probs", p.get("top_logprobs", []))][:5]}


def t_tokenize():
    d = post("/tokenize", {"model": A.model, "content": "Type Hi on the LED matrix",
                           "with_pieces": True})
    toks = d["tokens"]
    return {"n": len(toks), "first": toks[:4]}


def t_apply_template():
    d = post("/apply-template", {"model": A.model, "messages": msgs()})
    return {"prompt_head": d["prompt"][:80]}


print(f"llama.cpp at {A.url}, model {A.model}\n")
check("GET /v1/models", t_models)
check("chat completion (cold, loads model)", t_plain)
check("prompt cache reuse (timings.cache_n)", t_cache)
check("response_format json_schema", t_schema)
check("logprobs + top_logprobs=5 (chat)", t_logprobs_chat)
check("streaming SSE", t_stream)
check("stream + json_schema + logprobs", t_stream_schema_logprobs)
check("/completion n_probs + post_sampling", t_completion_nprobs)
check("/tokenize with_pieces", t_tokenize)
check("/apply-template", t_apply_template)

print("\nSUMMARY_JSON " + json.dumps(results, default=str))
