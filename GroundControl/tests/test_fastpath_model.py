"""The fast-path network's hand-written backward pass must match numerical gradients."""
import numpy as np

from fastpath.model import Net, features


def test_backward_matches_numerical_gradient():
    net = Net(n_out=4, seed=1)
    ids = [features("show a heart"), features("turn left 90 degrees"), features("what time is it")]
    y = np.array([1, 2, 3])
    _, g = net.loss_and_grads(ids, y, weight_decay=0)
    rng = np.random.default_rng(0)
    for name in ("W1", "b1", "W2", "b2", "E"):
        P = net.params()[name]
        idx = [tuple(rng.integers(0, s) for s in P.shape) for _ in range(5)]
        if name == "E":
            idx = [(ids[0][0], 3), (ids[1][2], 7)]
        for i in idx:
            old = P[i]
            P[i] = old + 1e-3
            lp, _ = net.loss_and_grads(ids, y, weight_decay=0)
            P[i] = old - 1e-3
            lm, _ = net.loss_and_grads(ids, y, weight_decay=0)
            P[i] = old
            num = (lp - lm) / 2e-3
            assert abs(num - g[name][i]) < 2e-3 + 2e-2 * abs(num), (name, i, num, g[name][i])


def test_features_are_stable_and_typo_tolerant():
    a, b = set(features("show a heart")), set(features("show a hearrt"))
    assert features("show a heart") == features("Show a HEART!")
    assert len(a & b) / len(a | b) > 0.4                      # a typo shares most features
