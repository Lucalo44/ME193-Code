"""Folder dataset + augmentation.

Images live in data/<task>/<label>/<session>_<n>.jpg. Every press of SPACE in collect_data.py
starts a new session.

Split: the last VAL_FRAC of *every* session goes to val, with a GAP of frames thrown away at
the boundary. That way every scene/lighting setup of every label is in training, and val is
still a separate stretch of time instead of the neighbors of training frames.
(Holding out whole sessions sounds stricter, but with two sessions per label it left each label
with one background in training, and the net learned the background instead of the fingers.)
"""
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from preprocess import TO_ARRAY


def augment(img, rng, flip):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-20, 20), rng.uniform(0.85, 1.15))
    M[:, 2] += rng.uniform(-0.08, 0.08, 2) * w
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    # lighting/white balance: the dim wall and the bright banner looked like different cameras
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img * rng.uniform(0.85, 1.15, 3)                                  # per-channel tint
    img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** rng.uniform(0.6, 1.6)  # gamma: dim <-> bright
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-25, 25))
    if rng.random() < 0.15:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if flip and rng.random() < 0.5:
        img = cv2.flip(img, 1)
    return img


class FolderDataset(Dataset):
    def __init__(self, items, task, train):
        self.items, self.task, self.train = items, task, train
        self.to_array = TO_ARRAY[task]
        # Flipping is fine for counting fingers, but a mirrored palm is a different palm.
        self.flip = task == "gesture"
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = cv2.imread(str(path))
        if self.train:
            img = augment(img, self.rng, self.flip)
        return torch.from_numpy(self.to_array(img)), label


def sessions_of(task, name):
    """{session_id: [files in recording order]} for one label."""
    sessions = defaultdict(list)
    for f in sorted((C.DATA_DIR / task / name).glob("*.jpg")):
        sessions[f.stem.split("_")[0]].append(f)
    return dict(sessions)


def load_split(task, val_frac=0.2, gap=5):
    """Returns (train_items, val_items, counts); items are (path, class_index).

    The last val_frac of each session is val; `gap` frames before it are dropped so the two
    sides of the boundary aren't near-duplicates.
    """
    train, val, counts = [], [], {}
    for idx, name in enumerate(C.CLASSES[task]):
        sessions = sessions_of(task, name)
        counts[name] = sum(len(fs) for fs in sessions.values())
        for fs in sessions.values():
            n_val = max(1, int(round(val_frac * len(fs)))) if len(fs) >= 10 else 0
            cut = len(fs) - n_val
            train += [(f, idx) for f in fs[:max(0, cut - gap)]]
            val += [(f, idx) for f in fs[cut:]] if n_val else []
    return train, val, counts
