"""Small, interpretable classifiers in numpy and scipy (no new dependency).

Settings were fixed before any evaluation and are not tuned:

* ``Majority``: always the most frequent answer of the training data; its probabilities
  are the training frequencies. The floor every model must clear.
* ``Logistic``: multinomial logistic regression with an L2 penalty, C = 1.0 (penalty
  ``0.5 * |W|^2`` against ``C`` times the class-weighted log loss), fitted with L-BFGS
  from zero weights, so the result is deterministic.
* ``Forest``: a random forest of 300 classification trees, each grown on a bootstrap
  sample with weighted Gini impurity, ``sqrt(features)`` candidates per split, at least 3
  examples per leaf, no depth limit, from a fixed seed. Probabilities are the mean of the
  trees' weighted leaf distributions.

Class weights are ``n / (k * n_class)`` from the **training fold only** (``k`` answers
present there), so a rare answer counts as much as a common one. ``Preprocess`` is fitted on
the training fold only: per-feature median for unknown values, an indicator column for every
feature that had an unknown value in training, then centring and scaling.
"""

import numpy as np
from scipy.optimize import minimize

C = 1.0
TREES, LEAF = 300, 3


def class_weights(y, classes):
    """Per-class weight from these labels only: ``n / (k * count)``; 0 for an absent class."""
    counts = np.bincount(y, minlength=classes).astype(float)
    present = counts > 0
    out = np.zeros(classes)
    out[present] = len(y) / (present.sum() * counts[present])
    return out


class Preprocess:
    """Median imputation, missing-value indicators and scaling, learned from training data."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        missing = np.isnan(X)
        with np.errstate(all="ignore"):
            self.median = np.nanmedian(np.where(missing.all(axis=0), 0.0, X), axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        self.indicators = np.flatnonzero(missing.any(axis=0))
        filled = self._fill(X)
        self.mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        return self

    def _fill(self, X):
        X = np.asarray(X, dtype=float)
        missing = np.isnan(X)
        filled = np.where(missing, self.median, X)
        return np.hstack((filled, missing[:, self.indicators].astype(float)))

    def transform(self, X):
        return (self._fill(X) - self.mean) / self.scale


class Majority:
    name = "majority"

    def fit(self, X, y, weights, classes):
        counts = np.bincount(y, minlength=classes)
        self.prior = counts / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.prior, (len(X), 1))


class Logistic:
    name = "logistic"

    def __init__(self, c=C):
        self.c = c

    def fit(self, X, y, weights, classes):
        n, d = X.shape
        onehot = np.eye(classes)[y]
        sample = weights[y]

        def loss(theta):
            W, b = theta[: d * classes].reshape(d, classes), theta[d * classes :]
            z = X @ W + b
            z -= z.max(axis=1, keepdims=True)
            log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            value = 0.5 * np.sum(W**2) - self.c * np.sum(sample * (onehot * log_p).sum(axis=1))
            residual = self.c * sample[:, None] * (np.exp(log_p) - onehot)
            gradient = np.concatenate(((W + X.T @ residual).ravel(), residual.sum(axis=0)))
            return value, gradient

        found = minimize(loss, np.zeros(d * classes + classes), jac=True, method="L-BFGS-B",
                         options={"maxiter": 5000, "gtol": 1e-8})
        self.converged = bool(found.success)
        self.W = found.x[: d * classes].reshape(d, classes)
        self.b = found.x[d * classes :]
        return self

    def predict_proba(self, X):
        z = X @ self.W + self.b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


class Tree:
    """One weighted-Gini classification tree, stored as flat arrays."""

    def __init__(self, X, Y, rows, rng, candidates, leaf):
        self.feature, self.threshold, self.left, self.right, self.value = [], [], [], [], []
        stack = [(rows, None, None)]
        while stack:
            rows, parent, side = stack.pop()
            node = len(self.feature)
            if parent is not None:
                (self.left if side == "left" else self.right)[parent] = node
            counts = Y[rows].sum(axis=0)
            self.feature.append(-1)
            self.threshold.append(0.0)
            self.left.append(-1)
            self.right.append(-1)
            self.value.append(counts / counts.sum())
            split = self._split(X, Y, rows, counts, rng, candidates, leaf)
            if split is None:
                continue
            feature, threshold = split
            self.feature[node], self.threshold[node] = feature, threshold
            goes_left = X[rows, feature] <= threshold
            stack.append((rows[~goes_left], node, "right"))
            stack.append((rows[goes_left], node, "left"))
        self.feature = np.array(self.feature)
        self.threshold = np.array(self.threshold)
        self.left, self.right = np.array(self.left), np.array(self.right)
        self.value = np.array(self.value)

    @staticmethod
    def _split(X, Y, rows, counts, rng, candidates, leaf):
        n = len(rows)
        total = counts.sum()
        if n < 2 * leaf or np.count_nonzero(counts) <= 1:
            return None
        impurity = total - np.sum(counts**2) / total
        best, found = impurity - 1e-12, None
        positions = np.arange(1, n)
        enough = (positions >= leaf) & (n - positions >= leaf)
        for feature in rng.choice(X.shape[1], candidates, replace=False):
            x = X[rows, feature]
            order = np.argsort(x, kind="stable")
            xs = x[order]
            left = np.cumsum(Y[rows][order], axis=0)[:-1]
            right = counts - left
            wl, wr = left.sum(axis=1), right.sum(axis=1)
            valid = enough & (xs[1:] > xs[:-1]) & (wl > 0) & (wr > 0)
            if not valid.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                score = (wl - np.sum(left**2, axis=1) / wl) + (wr - np.sum(right**2, axis=1) / wr)
            score = np.where(valid, score, np.inf)
            at = int(np.argmin(score))
            if score[at] < best:
                best, found = score[at], (int(feature), float((xs[at] + xs[at + 1]) / 2))
        return found

    def predict(self, X):
        node = np.zeros(len(X), dtype=int)
        while True:
            inner = self.feature[node] >= 0
            if not inner.any():
                return self.value[node]
            rows = np.flatnonzero(inner)
            here = node[rows]
            goes_left = X[rows, self.feature[here]] <= self.threshold[here]
            node[rows] = np.where(goes_left, self.left[here], self.right[here])


class Forest:
    name = "forest"

    def __init__(self, seed, trees=None, leaf=LEAF):
        self.seed, self.trees, self.leaf = seed, TREES if trees is None else trees, leaf

    def fit(self, X, y, weights, classes):
        rng = np.random.default_rng(self.seed)
        Y = np.eye(classes)[y] * weights[y][:, None]
        candidates = max(1, int(np.sqrt(X.shape[1])))
        self.forest = []
        for _tree in range(self.trees):
            rows = rng.integers(0, len(X), len(X))
            if Y[rows].sum() <= 0:
                continue
            self.forest.append(Tree(X, Y, rows, rng, candidates, self.leaf))
        return self

    def predict_proba(self, X):
        return np.mean([tree.predict(X) for tree in self.forest], axis=0)


def make(name, seed):
    """A fresh, unfitted model of the given kind."""
    if name == "majority":
        return Majority()
    if name == "logistic":
        return Logistic()
    if name == "forest":
        return Forest(seed)
    raise ValueError(f"unknown model {name!r}")


MODELS = ("majority", "logistic", "forest")
SETTINGS = {
    "majority": "most frequent answer of the training fold",
    "logistic": f"multinomial logistic regression, L2, C = {C}, class-weighted, L-BFGS from zero",
    "forest": f"random forest, {TREES} trees, bootstrap, weighted Gini, sqrt(features) per split, "
    f"at least {LEAF} examples per leaf, class-weighted, fixed seed",
}
