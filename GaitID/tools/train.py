#!/usr/bin/env python3
"""
Train both models on data/raw/*.csv and write them to models/.

    python tools/train.py                    # real recordings
    python tools/train.py --data data/synthetic
    python tools/train.py --axis az          # force Model 1's axis (default: best by CV)

Model 1: 1 feature (dynamic peak of one accel axis, log1p) -> Gaussian naive Bayes
Model 2: all IMU outputs (full_features)                    -> logistic regression
Scores are grouped cross-validation: every window is predicted by a model that
never saw any window from the same recording.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gaitid import config as C
from gaitid import data as D
from gaitid import features as F
from gaitid.model import Model


def peak_matrix(X, axis):
    return np.log1p(np.array([F.peak_feature(w, axis) for w in X]))


def full_matrix(X):
    return np.array([F.full_features(w) for w in X])


def cv_predict(est, feats, y, groups):
    folds = GroupKFold(n_splits=max(2, min(5, len(set(groups)))))
    return cross_val_predict(est, feats, y, groups=groups, cv=folds)


def report(name, y, pred, outdir):
    acc = float(np.mean(y == pred))
    print(f"\n{name}: grouped-CV accuracy {acc:.1%}")
    cm = confusion_matrix(y, pred, labels=range(len(C.CLASSES)))
    print("            " + " ".join(f"{c:>6}" for c in C.CLASSES))
    for c, row in zip(C.CLASSES, cm):
        print(f"  {c:>8}  " + " ".join(f"{v:6d}" for v in row))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=C.CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} ({acc:.0%})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"confusion_{name.split()[0].lower()}.png"), dpi=120)
    plt.close(fig)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "models"))
    ap.add_argument("--axis", default=C.PEAK_AXIS, choices=["auto"] + C.CHANNELS[:3])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, g = D.load_dir(args.data)
    print(f"{len(X)} windows from {len(set(g))} recordings; per class:",
          {c: int((y == i).sum()) for i, c in enumerate(C.CLASSES)})
    missing = [c for i, c in enumerate(C.CLASSES) if not (y == i).any()]
    if missing:
        raise SystemExit(f"No data for: {missing}")

    # ---- Model 1: one accel axis, peak only -----------------------------------
    axes = C.CHANNELS[:3] if args.axis == "auto" else [args.axis]
    scores = {}
    for a in axes:
        scores[a] = float(np.mean(cv_predict(GaussianNB(), peak_matrix(X, a), y, g) == y))
        print(f"  Model 1 axis {a}: {scores[a]:.1%}")
    axis = max(scores, key=scores.get)
    f1 = peak_matrix(X, axis)
    acc1 = report(f"Model1 peak-{axis}", y, cv_predict(GaussianNB(), f1, y, g), args.out)
    nb = GaussianNB().fit(f1, y)
    Model({"kind": "gaussian_nb", "input": "peak", "axis": axis, "classes": C.CLASSES,
           "theta": nb.theta_.tolist(), "var": nb.var_.tolist(),
           "prior": nb.class_prior_.tolist(), "cv_accuracy": acc1}).save(
        os.path.join(args.out, "model1_peak.json"))

    # ---- Model 2: all IMU outputs ---------------------------------------------
    f2 = full_matrix(X)
    est = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    acc2 = report("Model2 full-IMU", y, cv_predict(est, f2, y, g), args.out)
    est.fit(f2, y)
    sc, lr = est[0], est[1]
    Model({"kind": "logreg", "input": "full", "classes": C.CLASSES,
           "features": F.full_feature_names(),
           "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
           "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
           "cv_accuracy": acc2}).save(os.path.join(args.out, "model2_full.json"))

    print(f"\nSaved models to {args.out}.  Model 1 {acc1:.1%} vs Model 2 {acc2:.1%}")
    print("Next: python tools/sync_app.py  (copies models into app/ for the board)")


if __name__ == "__main__":
    main()
