"""Target functions the A5.2 spy test runs under recording (see ``privacy_bootstrap``).

``fold_generation`` and ``report_stage`` are the code under test; the ``inject_*`` functions
make deliberately prohibited accesses, so the positive controls can show they are caught.
"""

import json
import os
import subprocess
from pathlib import Path

from soundfile import info as soundfile_info  # imported before any wrapper could see it

from gmv.character.research.presence_model import b0check, enginepass, foldgen, gateway, integrity


def fold_generation(root, out, weights, expected):
    records = gateway.Gateway(root, step=3).records()
    cache = {}

    def rate_class(rate):
        if rate not in cache:
            cache[rate] = b0check.classify(rate, weights)["classification"]
        return cache[rate]

    foldgen.generate(records, root, out, rate_class, enginepass.quiet_mask, expected)


def report_stage(root, out, seal, vocabulary):
    gw = gateway.Gateway(root, step=6)
    tokens = gw.release_tokens(seal, gateway.Vocabulary(tuple(vocabulary["a3"]), tuple(vocabulary["candidate"])),
                               require_approved=False)
    groups = {r["id"]: r["group"] for r in gw.records()}
    integrity.write_artifact(out, "breakdown", "selection.json", gateway.breakdown(tokens, groups))


def inject_open(root, path):
    gateway.Gateway(root, step=3)
    with open(path, encoding="utf-8") as file:
        file.read(1)


def inject_probes(root, paths):
    gateway.Gateway(root, step=3)
    for path in paths:
        Path(path).exists()
        if os.path.isdir(path):
            os.listdir(path)


def inject_native(root, path):
    gateway.Gateway(root, step=3)
    soundfile_info(path)


def inject_child(root, path):
    gateway.Gateway(root, step=3)
    subprocess.run(["cat", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def raise_after_reading(root):
    """An error path: the target fails after the gateway has loaded its records."""
    records = gateway.Gateway(root, step=3).records()
    raise RuntimeError(json.dumps({"loaded": len(records)}))
