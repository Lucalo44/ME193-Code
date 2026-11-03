"""Folder dataset + augmentation.

Images live in data/<task>/<label>/<session>_<n>.jpg. Every press of SPACE in collect_data.py
starts a new session, and train/val are split by session, so the val set isn't just the
neighboring frames of the training set (which would make the accuracy look far too good).
"""
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from preprocess import TO_ARRAY


def augment(img, rng, flip):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.9, 1.1))
    M[:, 2] += rng.uniform(-0.06, 0.06, 2) * w
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-25, 25))
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


def load_split(task, val_frac=0.2, seed=0):
    """Returns (train_items, val_items, counts); items are (path, class_index)."""
    rnd = random.Random(seed)
    train, val, counts = [], [], {}
    for idx, name in enumerate(C.CLASSES[task]):
        files = sorted((C.DATA_DIR / task / name).glob("*.jpg"))
        counts[name] = len(files)
        if not files:
            continue
        sessions = defaultdict(list)
        for f in files:
            sessions[f.stem.split("_")[0]].append(f)
        keys = list(sessions)
        rnd.shuffle(keys)
        if len(keys) >= 2:
            held, n_held = [], 0
            for k in keys[:-1]:                      # always keep at least one session to train on
                if n_held >= val_frac * len(files):
                    break
                held.append(k)
                n_held += len(sessions[k])
            val_files = [f for k in held for f in sessions[k]]
            train_files = [f for k in keys if k not in held for f in sessions[k]]
        else:                                        # one session only: fall back to a random split
            rnd.shuffle(files)
            n_val = max(1, int(val_frac * len(files)))
            val_files, train_files = files[:n_val], files[n_val:]
        train += [(f, idx) for f in train_files]
        val += [(f, idx) for f in val_files]
    return train, val, counts
