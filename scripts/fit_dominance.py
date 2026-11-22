"""Fit the dominance coefficients (``gmv.visualizer.dominance.MODELS``).

    python scripts/fit_dominance.py frames.npz

``frames.npz`` comes from ``gather_dominance_frames.py``. Per instrument, a logistic model
of "this instrument's stem is the loudest of the four, by 3 dB or more" from: a constant,
its sustained activity (level, or energy for bass, averaged over 0.5 s), its transient
activity (attacks or pulses, each dying away over 0.5 s, capped at 4), and the mix's energy
share below 250 Hz and between 250 Hz and 4 kHz (averaged over 0.5 s). Wins and losses
weigh the same in each fit, so every instrument has the same chance beforehand. Printed:
accuracy with the songs split in three and each third held out in turn, then the
coefficients of the fit on all songs, which are the ones shipped.

The shipped coefficients are from the first 60 song folders, of which 59 were usable.
"""

import json
import sys

import numpy as np
from scipy.optimize import minimize

KEYS = ("vocals", "guitar", "bass", "drums")
d = dict(np.load(sys.argv[1]))
n = len(d["times"])
song = d["song"].astype(int)
TAU = 0.5


def ema(values, impulses=False):
    out, state, previous = np.zeros(n), 0.0, None
    keep = np.exp(-0.02 / TAU)
    for i in range(n):
        if song[i] != previous:
            state, previous = (0.0 if impulses else values[i]), song[i]
        state = state * keep + values[i] if impulses else values[i] + (state - values[i]) * keep
        out[i] = state
    return out


bands = np.stack([d[f"band_{i}"] for i in range(7)], axis=1)
total = bands.sum(axis=1) + 1e-12
shares = [ema(bands[:, :2].sum(axis=1) / total), ema(bands[:, 2:5].sum(axis=1) / total), ema(bands[:, 5:].sum(axis=1) / total)]
weight, features = {}, {}
for key in KEYS:
    drive = np.where(np.isfinite(d[f"{key}_energy"]), d[f"{key}_energy"], d[f"{key}_level"])
    hits = np.where(np.isfinite(d[f"{key}_energy"]), d[f"{key}_pulse"], d[f"{key}_attack"])
    weight[key] = np.nan_to_num(d[f"{key}_present"], nan=1.0)
    features[key] = np.stack([np.ones(n), ema(drive), np.minimum(ema(hits, True), 4.0), shares[0], shares[1]], axis=1)

true = np.stack([d[f"{k}_true_db"] for k in KEYS], axis=1)
order = np.sort(true, axis=1)
clear = (order[:, -1] - order[:, -2] >= 3.0) & (order[:, -1] > -45.0)
truth = true.argmax(axis=1)


def fit(mask, index, key):
    x, y = features[key][mask], (truth[mask] == index).astype(float)

    # Each instrument is given the same chance beforehand: wins and losses weigh the same
    # in the fit, so a score is evidence from the signals, not how often that instrument
    # happens to be the loudest in this collection.
    share = y.mean()
    each = np.where(y > 0, 0.5 / max(share, 1e-9), 0.5 / max(1 - share, 1e-9))

    def loss(w):
        z = x @ w
        return float(np.mean(each * (np.logaddexp(0, z) - y * z)) + 1e-3 * np.dot(w[1:], w[1:]))

    return minimize(loss, np.zeros(x.shape[1]), method="L-BFGS-B").x


def scores(models, weighted=True):
    return np.stack(
        [1 / (1 + np.exp(-(features[k] @ models[k]))) * (weight[k] if weighted else 1.0) for k in KEYS], axis=1
    )


def report(sc, mask, label):
    m = clear & mask & (sc.max(axis=1) > 0.02)
    pick = sc.argmax(axis=1)
    recall = {k: round(float((pick[m & (truth == i)] == i).mean()), 2) for i, k in enumerate(KEYS)}
    picked = {k: round(float((pick[m] == i).mean()), 2) for i, k in enumerate(KEYS)}
    print(f"{label:34s} accuracy {(pick[m] == truth[m]).mean():.3f}  mean recall {np.mean(list(recall.values())):.3f}  recall {recall}  picked {picked}")
    return float((pick[m] == truth[m]).mean())


folds = []
for fold in (0, 1, 2):
    train = clear & (song % 3 != fold)
    models = {k: fit(train, i, k) for i, k in enumerate(KEYS)}
    folds.append(report(scores(models), song % 3 == fold, f"held-out third {fold}"))
print("cross-validated accuracy", round(float(np.mean(folds)), 3), "| chance 0.25 | always the commonest", round(float(max((truth[clear] == i).mean() for i in range(4))), 3))
models = {k: fit(clear, i, k) for i, k in enumerate(KEYS)}
report(scores(models), np.ones(n, bool), "fitted on all (optimistic)")
print("true winner shares", {k: round(float((truth[clear] == i).mean()), 2) for i, k in enumerate(KEYS)}, "clear frames", int(clear.sum()), "of", n, "songs", len(set(song.tolist())))
sc = scores(models)
for i, k in enumerate(KEYS):
    print(k, "score percentiles when it wins", np.round(np.percentile(sc[clear & (truth == i), i], [10, 50, 90]), 2).tolist(), "when it loses", np.round(np.percentile(sc[clear & (truth != i), i], [10, 50, 90]), 2).tolist())
print(json.dumps({k: [round(float(v), 3) for v in w] for k, w in models.items()}))
