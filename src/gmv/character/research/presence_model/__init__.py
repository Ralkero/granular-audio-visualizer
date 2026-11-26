"""Phase A5.2 research harness: comparing vocal-presence detectors (offline only).

The registered protocol is ``docs/VOCAL_PRESENCE_MODEL_PROTOCOL.md``. This package holds
the parts that need only NumPy and SciPy:

* ``frontend``: model C's log-mel front end on the analysis engine's frame grid (§4);
* ``labels``: frame labels, boundary collars and the frames each use excludes (§6, §10);
* ``weights``: the equal-group, class-balanced training objective and its exact batching (§6);
* ``folds``: the class-aware, capacity-limited inner folds (§9);
* ``postprocess``: the silence gate and hysteresis shared by every arm (§8);
* ``metrics``: per-group recall and false vocal activity, group averages, events (§10-§11);
* ``screening``: the low-denominator rule and the screening classes (§12.1, §13, Amendment 1);
* ``bootstrap``: descriptive group-bootstrap intervals that always carry their caveat (§12);
* ``seeds``: the registered seeds (§7).

The convolutional network itself needs PyTorch and lives in ``experiments/presence``, in
the separate experiments environment. The application never imports this package.
"""
