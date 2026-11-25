"""FakeLLM: same interface as LlmClient, canned plans, fake token stream with
plausible top-5 alternatives. Lets the whole app, the dashboard and the tests
run on a laptop with no model. `delay_s` per token mimics the board (~0.23 s)."""
import json
import random
import re
import time

CANNED = [
    (r"yaw|heading", {"parts": ["what is your yaw? reply on luca"], "steps": [
        {"skill": "mqtt_publish", "args": {"topic": "luca", "message": "yaw is {imu.yaw} deg"}}]}),
    (r"danc", {"parts": ["dance with the lego bot", "tell me how in sync you were on luca"], "steps": [{"skill": "dance", "args": {"bpm": 110, "sections": [
        {"bars": 2, "mode": "mirror", "car": "shimmy", "lego": "wave"},
        {"bars": 1, "mode": "call_response", "car": "spin", "lego": "clap"},
        {"bars": 1, "mode": "unison", "car": "bow", "lego": "bow"}]}},
        {"skill": "mqtt_publish", "args": {"topic": "luca", "message": "bow complete, max error {dance.max_sync_error_ms} ms"}}]}),
    (r"heart", {"parts": ["show a heart"], "steps": [{"skill": "matrix_icon", "args": {"name": "heart"}}]}),
    (r"\bhi\b", {"parts": ["Type Hi on the LED matrix", "send back on topic luca 'I am done'"], "steps": [
        {"skill": "matrix_text", "args": {"text": "Hi"}},
        {"skill": "mqtt_publish", "args": {"topic": "luca", "message": "I am done"}}]}),
    (r"weather|temperature", {"think": "lookup weather", "steps": [{"skill": "lookup", "args": {"what": "weather", "about": ""}}]}),
    (r"strawberry", {"think": "compute count", "steps": [{"skill": "compute", "args": {"op": "count", "text": "strawberry", "of": "e"}}]}),
    (r"^remember", {"think": "memory save", "steps": [{"skill": "memory", "args": {"op": "save", "key": "favorite color", "value": "green"}}]}),
    (r"favou?rite color\?", {"think": "memory recall", "steps": [{"skill": "memory", "args": {"op": "recall", "key": "favorite color", "value": ""}}]}),
    (r"2\s*\+\s*2", {"think": "answer", "steps": [{"skill": "answer", "args": {"text": "2 + 2 is 4."}}]}),
    (r"what board|plugged|pins", {"think": "self_report board, then self_report pins", "steps": [
        {"skill": "self_report", "args": {"topic": "board"}}, {"skill": "self_report", "args": {"topic": "pins"}}]}),
    (r"spin", {"parts": ["spin"], "steps": [{"skill": "spin", "args": {"times": 1}}]}),
]


def _split(text):
    return re.findall(r'[A-Za-z_]+|\s+|\d+|[^\sA-Za-z_\d]{1,2}', text)


class FakeLLM:
    def __init__(self, delay_s=0.02, model="fake"):
        self.delay_s = delay_s
        self.model = model
        self.url = "fake://"

    def models(self):
        return [self.model]

    def tokenize(self, text):
        return [{"id": abs(hash(p)) % 50000, "piece": p} for p in _split(text)]

    def apply_template(self, messages):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + \
            "<|im_start|>assistant\n"

    def plan_for(self, text):
        for pat, plan in CANNED:
            if re.search(pat, text, re.I):
                return plan
        return {"parts": [], "steps": []}

    def chat_stream(self, messages, schema=None, max_tokens=200, on_token=None, temperature=0.0,
                    top_k=5, cancel=None):
        cmd = messages[-1]["content"]
        if schema is None and cmd.startswith("Facts: "):           # answer composer
            facts = cmd[len("Facts: "):].split("\nQuestion: ")[0]
            text = f"Here's what I found: {facts}"
            toks = [{"i": i, "piece": p, "logprob": None, "p": 0.9, "top": [{"piece": p, "p": 0.9}], "t_ms": 1}
                    for i, p in enumerate(_split(text))]
            for t in toks:
                if on_token:
                    on_token(t)
            return {"text": text, "tokens": toks, "timings": {}, "wall_ms": 1, "first_token_ms": 1}
        plan = self.plan_for(cmd)
        text = json.dumps(plan, separators=(",", ":"))
        t0 = time.time()
        toks = []
        for i, piece in enumerate(_split(text)):
            if cancel is not None and cancel.is_set():
                break
            time.sleep(self.delay_s)
            p = random.uniform(0.55, 0.99)
            alts = ["matrix", "mqtt", "dance", "drive", "spin", "read", " ", '"']
            top = [{"piece": piece, "p": round(p, 3)}] + [
                {"piece": a, "p": round((1 - p) / (k + 2), 3)} for k, a in enumerate(random.sample(alts, 4))]
            tok = {"i": i, "piece": piece, "logprob": None, "p": round(p, 3), "top": top,
                   "t_ms": round((time.time() - t0) * 1000)}
            toks.append(tok)
            if on_token:
                on_token(tok)
        n = len(toks)
        return {"text": "".join(t["piece"] for t in toks), "tokens": toks,
                "timings": {"prompt_n": 12, "cache_n": 380, "prompt_ms": 900, "predicted_n": n,
                            "predicted_ms": n * self.delay_s * 1000,
                            "predicted_per_second": 1 / self.delay_s if self.delay_s else 0,
                            "prompt_per_second": 11.0},
                "wall_ms": round((time.time() - t0) * 1000), "first_token_ms": round(self.delay_s * 1000)}

    def chat_tools(self, messages, tools, max_tokens=200, temperature=0.0):
        return {"message": {"role": "assistant", "content": "ok"}, "timings": {}, "wall_ms": 1}
