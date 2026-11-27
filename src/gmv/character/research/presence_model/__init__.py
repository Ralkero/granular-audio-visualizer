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

Step 1 (synthetic only; readiness proposal revision 4.4, provisional until Amendment 3):

* ``seedplan``: every production seed identity, per-consumer streams, collision checks,
  the deterministic remapping of gate and sensitivity seeds, and the replay ledger;
* ``b0check``: the four-level check that B0 runs at a sample rate without a silent fallback;
* ``grid``, ``enginepass``: the canonical segment and frame grid, the engine's quiet rule and
  B1's features;
* ``integrity``: integrity records and artifact-tree comparison;
* ``foldgen``: outer and inner fold generation with an independent verifier;
* ``gateway``: Tier 2 fields only, seal-gated tokens and a parser with no fixed vocabulary;
* ``timing``: exec-to-exit accounting, the projection and the scheduling simulations;
* ``normalise``, ``augment``: normalisation interfaces and model C's augmentation.

The convolutional network itself needs PyTorch and lives in ``experiments/presence``, in
the separate experiments environment. The application never imports this package.
"""
