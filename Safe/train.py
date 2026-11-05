"""Train either net from the images in data/.

    python3 train.py --task gesture
    python3 train.py --task owner --epochs 40

Saves the best-on-validation weights to models/<task>.pt and prints a confusion matrix.
For the owner net it also prints how often a not_me hand would clear OWNER_THRESH
(the false-accept rate, which is the number that matters for a lock).
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
import models
from dataset import FolderDataset, load_split


def evaluate(model, loader, device, n_classes):
    model.eval()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).softmax(1).cpu()
            for t, q in zip(y, p.argmax(1)):
                conf[t, q] += 1
            probs.append(p)
            labels.append(y)
    acc = conf.diag().sum().item() / max(1, conf.sum().item())
    return acc, conf, torch.cat(probs), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gesture", "owner"], default="gesture")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    classes = C.CLASSES[a.task]
    train_items, val_items, counts = load_split(a.task)
    print("images per class:", counts)
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise SystemExit(f"No images for {missing}. Record them with collect_data.py --task {a.task}.")
    print(f"train {len(train_items)}  val {len(val_items)}")

    device = models.pick_device()
    model = models.build(a.task).to(device)
    train_dl = DataLoader(FolderDataset(train_items, a.task, train=True), a.batch, shuffle=True)
    val_dl = DataLoader(FolderDataset(val_items, a.task, train=False), a.batch)

    # weight classes by inverse frequency so a big "none" or "not_me" pile doesn't dominate
    n_train = torch.bincount(torch.tensor([y for _, y in train_items]), minlength=len(classes)).float()
    weights = (n_train.sum() / (len(classes) * n_train.clamp(min=1))).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(train_dl))

    C.MODEL_DIR.mkdir(exist_ok=True)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        acc, *_ = evaluate(model, val_dl, device, len(classes))
        flag = ""
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "task": a.task, "classes": classes,
                        "val_acc": acc}, models.model_path(a.task))
            flag = "  * saved"
        print(f"epoch {ep:3d}  loss {total / n:.3f}  val acc {acc:.3f}  ({time.time() - t0:.1f}s){flag}")

    model = models.load(a.task, device)
    acc, conf, probs, labels = evaluate(model, val_dl, device, len(classes))
    print(f"\nbest val acc {acc:.3f}  ->  {models.model_path(a.task)}")
    print("confusion (rows = true, cols = predicted):")
    print("        " + " ".join(f"{c:>7}" for c in classes))
    for c, row in zip(classes, conf):
        print(f"{c:>7} " + " ".join(f"{v:7d}" for v in row.tolist()))

    # Per-session accuracy: one session far below the rest usually means the net learned
    # that session's background/lighting instead of the hand. Record that label again elsewhere.
    print("\nval accuracy per recording session:")
    pred = probs.argmax(1)
    by_session = {}
    for (path, y), q in zip(val_items, pred.tolist()):
        key = (classes[y], path.stem.split("_")[0])
        hit, n = by_session.get(key, (0, 0))
        by_session[key] = (hit + (q == y), n + 1)
    for (name, sess), (hit, n) in sorted(by_session.items()):
        print(f"  {name:>7}  session {sess}  {hit / n:6.1%}  ({n} imgs)" + ("   <-- check this one" if hit / n < 0.6 else ""))

    if a.task == "owner":
        p_me = probs[:, classes.index("me")]
        is_me = labels == classes.index("me")
        far = (p_me[~is_me] >= C.OWNER_THRESH).float().mean().item() if (~is_me).any() else float("nan")
        frr = (p_me[is_me] < C.OWNER_THRESH).float().mean().item() if is_me.any() else float("nan")
        print(f"\nper-frame at OWNER_THRESH={C.OWNER_THRESH}: false accept {far:.1%}, false reject {frr:.1%}")
        print("(the safe averages P(me) over each digit's frames, so the real rates are lower)")


if __name__ == "__main__":
    main()
