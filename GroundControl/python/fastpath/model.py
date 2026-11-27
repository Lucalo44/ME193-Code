"""The fast-path intent classifier (Part 2, A3): a small fastText-style network, in numpy.

    text -> hashed features -> embedding bag (mean) -> Linear+ReLU -> Linear -> softmax

Features are word unigrams, word bigrams and character trigrams (with word boundaries),
each hashed into one of N_BUCKETS rows of the embedding table. Hashing means no vocabulary
file and no unknown words: a typo still shares most of its character trigrams with the word
it meant to be.

Written from scratch (forward and backward pass by hand) so the same file trains on the
laptop and runs on the board, which already has numpy: no PyTorch, no onnxruntime.
Inference is two small matrix products, well under a millisecond.
"""
import re
import zlib

import numpy as np

N_BUCKETS = 4096
DIM = 32
HIDDEN = 64

INTENTS = [
    "none",
    "show_icon", "write_text", "clear_matrix",
    "turn", "spin", "drive", "stop",
    "dance_style",
    "self_board", "self_pins", "self_devices", "self_software", "self_health", "self_network", "self_all",
    "lookup_time", "lookup_date", "lookup_weather", "lookup_forecast",
    "memory_recall",
]


def features(text):
    """-> sorted unique bucket ids for the text."""
    t = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower())
    words = t.split()
    feats = [f"w:{w}" for w in words] + [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    for w in words:
        p = f"<{w}>"
        feats += [f"c:{p[i:i + 3]}" for i in range(len(p) - 2)]
    return sorted({zlib.crc32(f.encode()) % N_BUCKETS for f in feats}) or [0]


class Net:
    def __init__(self, n_out=len(INTENTS), seed=0):
        rng = np.random.default_rng(seed)
        self.E = (rng.standard_normal((N_BUCKETS, DIM)) * 0.1).astype(np.float32)
        self.W1 = (rng.standard_normal((DIM, HIDDEN)) * np.sqrt(2 / DIM)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, np.float32)
        self.W2 = (rng.standard_normal((HIDDEN, n_out)) * np.sqrt(1 / HIDDEN)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)

    def params(self):
        return {"E": self.E, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def n_params(self):
        return sum(v.size for v in self.params().values())

    # ---- inference
    def embed(self, ids_list):
        return np.stack([self.E[ids].mean(0) for ids in ids_list])

    def forward(self, ids_list):
        x = self.embed(ids_list)
        h = np.maximum(0, x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True), (x, h)

    def predict(self, text):
        p, _ = self.forward([features(text)])
        return p[0]

    # ---- training (cross-entropy, Adam, hand-written backward pass)
    def loss_and_grads(self, ids_list, y, weight_decay=1e-5):
        p, (x, h) = self.forward(ids_list)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
        dz = p.copy()
        dz[np.arange(n), y] -= 1
        dz /= n                                          # d loss / d logits
        g = {"W2": h.T @ dz + weight_decay * self.W2, "b2": dz.sum(0)}
        dh = dz @ self.W2.T
        dh[h <= 0] = 0                                   # ReLU
        g["W1"] = x.T @ dh + weight_decay * self.W1
        g["b1"] = dh.sum(0)
        dx = dh @ self.W1.T                              # d loss / d mean embedding, per example
        dE = np.zeros_like(self.E)
        for i, ids in enumerate(ids_list):
            np.add.at(dE, ids, dx[i] / len(ids))          # the mean spreads the gradient over its rows
        g["E"] = dE
        return loss, g

    def save(self, path):
        np.savez_compressed(path, intents=np.array(INTENTS), **{k: v.astype(np.float16) for k, v in self.params().items()})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(n_out=len(d["intents"]))
        for k in ("E", "W1", "b1", "W2", "b2"):
            setattr(net, k, d[k].astype(np.float32))
        return net, [str(s) for s in d["intents"]]


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999):
        self.p, self.lr, self.b1, self.b2, self.t = params, lr, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
