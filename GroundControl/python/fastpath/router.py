"""The fast-path router (Part 2, A3): answer common one-step commands without the LLM.

    command -> normalize -> reflexes (status/stop/replay) -> FAST PATH -> LLM planner

The fast path takes a command only if ALL of these hold:
  1. the network's top intent has probability >= TAU and isn't "none";
  2. it beats the runner-up by >= MARGIN;
  3. the command has no sequencing, conjunction or negation markers (and, then, after, but,
     don't, not, unless, while, commas joining clauses...);
  4. every slot the intent needs parses (python/fastpath/parse.py).
Otherwise the LLM plans it, as before. TAU and MARGIN are calibrated on the dev data for
>= 99.5% precision on the commands the fast path accepts (tools/fastpath/train.py): a wrong
fast action is worse than a slow right one.
"""
import json
import re
import time
from pathlib import Path

from fastpath.model import Net
from fastpath.parse import build

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model.npz"
CALIBRATION = HERE / "calibration.json"

MARKERS = re.compile(r"\b(and|then|after|afterwards|before|but|don'?t|do not|doesn'?t|not|never|unless|while|"
                     r"until|also|plus|instead|except|without|if|when|so|or|nothing else|no )\b|[,;:]|\?.+\S|\.\s+\S",
                     re.I)


# Politeness and openers aren't clauses: "hey robot, what time is it, please?" is one command.
# They're stripped from the ends before the marker check (never from the middle). Not "right
# now": "turn to the right now" would lose its direction.
_LEAD = re.compile(r"^(?:\s*(?:hey|hi|yo|ok(?:ay)?|so|well|um|alright|quick question|hey robot|hey ground control|"
                   r"robot|ground control|please|pleease|pls|plz|now)\b[\s,:;!.-]*)+", re.I)
_TRAIL = re.compile(r"(?:[\s,;:-]*\b(?:please|pleease|pls|plz|thanks|thank you|thx|ty|real quick|now|for me|"
                    r"again|ok|okay)\b[\s.!?]*)+$", re.I)


def core_text(text):
    t = (text or "").strip()
    for _ in range(3):
        t2 = _TRAIL.sub("", _LEAD.sub("", t)).strip()
        if t2 == t:
            break
        t = t2
    return t or (text or "").strip()


class Router:
    def __init__(self, model=MODEL, calibration=CALIBRATION):
        self.net, self.intents = Net.load(model)
        cal = json.loads(Path(calibration).read_text()) if Path(calibration).exists() else {}
        self.tau = cal.get("tau", 0.97)
        self.margin = cal.get("margin", 0.9)

    def route(self, text):
        """-> {"fast": bool, "plan", "intent", "p", "margin", "probs": {intent: p}, "reason", "ms"}."""
        t0 = time.perf_counter()
        p = self.net.predict(text)
        order = p.argsort()[::-1]
        top, second = int(order[0]), int(order[1])
        intent = self.intents[top]
        out = {"fast": False, "plan": None, "intent": intent, "p": round(float(p[top]), 4),
               "margin": round(float(p[top] - p[second]), 4), "tau": self.tau, "margin_min": self.margin,
               "probs": {self.intents[i]: round(float(p[i]), 4) for i in order[:6]}}
        core = core_text(text)
        m = MARKERS.search(core)
        if intent == "none":
            out["reason"] = "intent: none"
        elif len(re.findall(r"[a-z0-9]+", core.lower())) < 2:
            out["reason"] = "too short to be sure"        # "mub": three letters of gibberish
        elif p[top] < self.tau:
            out["reason"] = f"p {p[top]:.2f} < {self.tau}"
        elif p[top] - p[second] < self.margin:
            out["reason"] = f"margin {p[top] - p[second]:.2f} < {self.margin}"
        elif m:
            out["reason"] = f"marker \"{m.group(0).strip()}\""
        else:
            steps = build(intent, text)
            if steps is None:
                out["reason"] = "slot didn't parse"
            else:
                out.update(fast=True, plan={"think": "", "steps": steps}, reason="fast path")
        out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out
