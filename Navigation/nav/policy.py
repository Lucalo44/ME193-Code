"""The behavior-cloned navigator: observation -> (next maneuver, steps until it).

That is what a GPS voice needs: "turn left in 30 m", then "turn left now" as the distance
reaches zero. A small MLP with two heads learns both from the demonstrations:
    maneuver head: softmax over MANEUVERS (left / right / stairs / arrive)
    distance head: log(1 + steps until that maneuver)
It's trained in PyTorch on the laptop (tools/train.py) and saved to .npz, so the board runs it
in numpy, the same way StepNet runs there.

explain() returns each input's contribution to the chosen maneuver (gradient x input, in
normalised units), which the laptop dashboard shows live.
"""
import json

import numpy as np


class Policy:
    def __init__(self, path=None):
        self.w = None
        self.meta = {}
        if path is not None:
            try:
                d = np.load(str(path), allow_pickle=False)
                self.w = {k: d[k] for k in d.files if k != "meta"}
                self.meta = json.loads(str(d["meta"]))
            except (OSError, KeyError, ValueError):
                self.w = None

    @property
    def loaded(self):
        return self.w is not None

    @property
    def maneuvers(self):
        return self.meta.get("maneuvers", ["left", "right", "stairs", "arrive"])

    @property
    def route_len(self):
        return float(self.meta.get("route_len", 400.0))

    def _forward(self, obs):
        w = self.w
        z = (np.asarray(obs, float) - w["mean"]) / w["scale"]
        a1 = w["W1"] @ z + w["b1"]
        h1 = np.maximum(a1, 0)
        a2 = w["W2"] @ h1 + w["b2"]
        h2 = np.maximum(a2, 0)
        logits = w["Wm"] @ h2 + w["bm"]
        logd = float((w["Wd"] @ h2 + w["bd"])[0])
        return z, a1, a2, logits, logd

    def act(self, obs):
        """Returns {maneuver, probs, steps} (steps = predicted steps until the maneuver)."""
        if not self.loaded:
            return None
        _, _, _, logits, logd = self._forward(obs)
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        k = int(np.argmax(p))
        return {"maneuver": self.maneuvers[k], "probs": p.tolist(), "steps": max(0.0, float(np.expm1(logd)))}

    def explain(self, obs):
        """Gradient x input of the winning maneuver's logit, per feature."""
        if not self.loaded:
            return None
        w = self.w
        z, a1, a2, logits, _ = self._forward(obs)
        k = int(np.argmax(logits))
        g2 = w["Wm"][k] * (a2 > 0)
        g1 = (w["W2"].T @ g2) * (a1 > 0)
        gz = w["W1"].T @ g1
        return (gz * z).tolist()


def export(state_dict, mean, scale, meta, path):
    """state_dict: torch weights as numpy (keys from tools/train.py's Net)."""
    np.savez(str(path), mean=mean, scale=scale,
             W1=state_dict["l1.weight"], b1=state_dict["l1.bias"],
             W2=state_dict["l2.weight"], b2=state_dict["l2.bias"],
             Wm=state_dict["man.weight"], bm=state_dict["man.bias"],
             Wd=state_dict["dist.weight"], bd=state_dict["dist.bias"],
             meta=json.dumps(meta))
