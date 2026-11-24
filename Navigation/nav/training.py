"""Turning walks into a trained navigator (laptop only: imports torch).

    prepare(walks)  replay each walk through the pipeline, clean its labels, build targets
    fit(prepared)   stairs classifier, then the policy MLP; returns (Policy, StairsClassifier)
    leave_one_out   train on all walks but one, test on that one, for every walk
"""
import copy

import numpy as np
import torch
import torch.nn as nn

import config as C
from nav.pipeline import replay
from nav.policy import Policy, export
from nav.stairs import StairsClassifier
from nav.walks import clean, targets


class Net(nn.Module):
    def __init__(self, n_in, hidden=C.HIDDEN, n_out=len(C.MANEUVERS)):
        super().__init__()
        self.l1, self.l2 = nn.Linear(n_in, hidden), nn.Linear(hidden, hidden)
        self.man, self.dist = nn.Linear(hidden, n_out), nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.l2(torch.relu(self.l1(x))))
        return self.man(h), self.dist(h).squeeze(-1)


def _blank_policy(route_len):
    p = Policy(None)
    p.meta = {"route_len": float(route_len)}
    return p


def prepare(walks, stairs=None, route_len=400.0):
    """walks: dicts from walks.load_walk(). Returns one dict per walk with records, labels, X, targets."""
    out = []
    for w in walks:
        recs = replay(w["imu"], policy=_blank_policy(route_len), stairs=stairs or StairsClassifier(None))
        labels = w["labels"] or clean(recs, w["taps"])
        man, dist, st = targets(labels, len(recs))
        n = min(len(man), len(recs))       # a walk cut off before arriving has its arrive label past the last step
        man, dist, st = man[:n], dist[:n], st[:n]
        out.append({"walk": w, "records": recs, "labels": labels, "n": n,
                    "X": np.array([r["obs"] for r in recs[:n]]).reshape(n, -1),
                    "SF": np.array([r["stairs_features"] for r in recs[:n]]).reshape(n, -1),
                    "man": man, "dist": dist, "stairs": st,
                    "weight": C.DAGGER_WEIGHT if w["meta"].get("mode") == "dagger" else 1.0})
    return out


def route_length(walks_prepared):
    lens = [p["n"] for p in walks_prepared if p["walk"]["meta"].get("mode") != "dagger" and p["n"]]
    lens = lens or [p["n"] for p in walks_prepared if p["n"]]
    return float(np.mean(lens)) if lens else 400.0


def reference_trail(walks_prepared, max_points=150):
    """Path of the demo whose length is closest to the median, for the phone map."""
    demos = [p for p in walks_prepared if p["n"]]
    if not demos:
        return []
    med = np.median([p["n"] for p in demos])
    best = min(demos, key=lambda p: abs(p["n"] - med))
    pts = [[r["state"]["x"], r["state"]["y"]] for r in best["records"][:best["n"]]]
    step = max(1, len(pts) // max_points)
    return [[round(x, 2), round(y, 2)] for x, y in pts[::step]] + [pts[-1]]


def fit_stairs(prepared):
    X = np.concatenate([p["SF"] for p in prepared]) if prepared else np.zeros((0, 6))
    y = np.concatenate([p["stairs"] for p in prepared]) if prepared else np.zeros(0)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return StairsClassifier(None), "not trained (fewer than 10 labelled stairs steps)"
    clf = StairsClassifier.fit(X, y)
    acc = float(np.mean([(clf.predict(x) > 0.5) == bool(t) for x, t in zip(X, y)]))
    return clf, f"trained on {int(y.sum())} stairs / {int((1 - y).sum())} flat steps, train accuracy {acc:.1%}"


def fit_policy(prepared, route_len, epochs=600, noise=0.05, seed=0, verbose=False):
    torch.manual_seed(seed)
    X = np.concatenate([p["X"] for p in prepared])
    man = np.concatenate([p["man"] for p in prepared])
    dist = np.concatenate([p["dist"] for p in prepared])
    wts = np.concatenate([np.full(p["n"], p["weight"]) for p in prepared])
    mean, scale = X.mean(0), X.std(0) + 1e-3
    Z = torch.tensor((X - mean) / scale, dtype=torch.float32)
    ym = torch.tensor(man, dtype=torch.long)
    yd = torch.tensor(np.log1p(dist), dtype=torch.float32)
    w = torch.tensor(wts, dtype=torch.float32)
    counts = np.bincount(man, minlength=len(C.MANEUVERS)).astype(float)
    cw = torch.tensor(np.where(counts > 0, counts.sum() / (len(C.MANEUVERS) * np.maximum(counts, 1)), 0),
                      dtype=torch.float32)
    net = Net(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=cw, reduction="none")
    for ep in range(epochs):
        net.train()
        logits, ld = net(Z + noise * torch.randn_like(Z))
        loss = ((ce(logits, ym) + 0.5 * (ld - yd) ** 2) * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            print(f"  epoch {ep:4d}  loss {loss.item():.3f}")
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    meta = {"maneuvers": C.MANEUVERS, "route_len": route_len, "features": X.shape[1],
            "walks": [str(p["walk"]["path"].name) for p in prepared], "steps": int(len(X))}
    return sd, mean, scale, meta


def fit(walks, epochs=600, verbose=False):
    """Full training: returns (policy_state, stairs_clf, report dict)."""
    first = prepare(walks)                                   # pass 1: labels + stairs model
    route_len = route_length(first)
    stairs, stairs_msg = fit_stairs(first)
    prepared = prepare(walks, stairs=stairs if stairs.trained else None, route_len=route_len)   # pass 2: final features
    sd, mean, scale, meta = fit_policy(prepared, route_len, epochs=epochs, verbose=verbose)
    meta["reference_trail"] = reference_trail(prepared)
    return (sd, mean, scale, meta), stairs, {"route_len": route_len, "stairs": stairs_msg, "prepared": prepared}


def to_policy(trained, path):
    sd, mean, scale, meta = trained
    export(sd, mean, scale, meta, path)
    return Policy(path)


def evaluate(policy, prepared_walk):
    """Per-step maneuver accuracy, distance error, and how well 'now' lines up with real maneuvers."""
    p = prepared_walk
    if p["n"] == 0:
        return None
    preds = [policy.act(x) for x in p["X"]]
    pm = np.array([C.MANEUVERS.index(d["maneuver"]) for d in preds])
    pd = np.array([d["steps"] for d in preds])
    acc = float(np.mean(pm == p["man"]))
    mae = float(np.mean(np.abs(pd - p["dist"])))
    near = p["dist"] <= 20                                   # the part of the walk where it matters
    acc_near = float(np.mean(pm[near] == p["man"][near])) if near.any() else float("nan")
    mae_near = float(np.mean(np.abs(pd[near] - p["dist"][near]))) if near.any() else float("nan")
    # for each real maneuver: first step the navigator said "<that maneuver> now", relative to it
    timing = []
    for m in p["labels"]["maneuvers"]:
        k, want = m["step"], C.MANEUVERS.index(m["maneuver"])
        lo = max(0, k - 25)
        hits = [i for i in range(lo, min(p["n"], k + 6)) if pm[i] == want and pd[i] <= C.NOW_STEPS]
        timing.append({"maneuver": m["maneuver"], "step": k, "called_at": (hits[0] - k) if hits else None})
    return {"acc": acc, "mae": mae, "acc_near": acc_near, "mae_near": mae_near, "timing": timing}


def leave_one_out(walks, epochs=600):
    results = []
    for i in range(len(walks)):
        train = walks[:i] + walks[i + 1:]
        trained, stairs, rep = fit(train, epochs=epochs)
        pol = Policy(None)
        sd, mean, scale, meta = trained
        pol.w = {"mean": mean, "scale": scale, "W1": sd["l1.weight"], "b1": sd["l1.bias"],
                 "W2": sd["l2.weight"], "b2": sd["l2.bias"], "Wm": sd["man.weight"], "bm": sd["man.bias"],
                 "Wd": sd["dist.weight"], "bd": sd["dist.bias"]}
        pol.meta = copy.deepcopy(meta)
        test = prepare([walks[i]], stairs=stairs if stairs.trained else None, route_len=rep["route_len"])[0]
        results.append({"walk": walks[i]["path"].name, **(evaluate(pol, test) or {})})
    return results
