"""Target functions the A5.2 spy test runs under recording (see ``privacy_bootstrap``).

``fold_generation`` and ``report_stage`` are the code under test; the ``inject_*`` functions
make deliberately prohibited accesses, so the positive and negative controls can show which
recorder catches them.
"""

import glob
import json
import os
import subprocess
from pathlib import Path

from soundfile import info as soundfile_info

from gmv.character.research.presence import identity
from gmv.character.research.presence_model import b0check, enginepass, foldgen, gateway, integrity, musdb


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


def inject_probes(root, audio):
    """PC4 as specified: a relative stat and listing from the project root, and an existence
    probe of validation audio."""
    gateway.Gateway(root, step=3)
    os.chdir(root)
    os.stat("final-test/annotations.jsonl")
    os.listdir("final-test")
    Path(audio).exists()


def inject_native(root, path):
    """PC5: libsndfile opens the file in native code (Python sees only soundfile's stat)."""
    gateway.Gateway(root, step=3)
    soundfile_info(path)


def inject_child(root, path):
    gateway.Gateway(root, step=3)
    subprocess.run(["cat", path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def inject_relative(root, rel, operation, directory="."):
    """A prohibited access by a relative path after changing directory (``directory`` is
    relative to ``root``)."""
    gateway.Gateway(root, step=3)
    os.chdir(os.path.join(root, directory))
    if operation == "open":
        with open(rel, encoding="utf-8") as file:
            file.read(1)
    elif operation == "stat":
        os.stat(rel)
    elif operation == "list":
        os.listdir(rel)
    elif operation == "glob":
        glob.glob(rel)
    elif operation == "child":
        subprocess.run(["cat", rel], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        raise ValueError("unknown operation")


def inject_unresolvable(root):
    """Accesses whose targets the Python recorders cannot establish: a listing and a stat
    through a directory descriptor."""
    gateway.Gateway(root, step=3)
    fd = os.open(os.path.join(root, "validation"), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.listdir(fd)
        os.stat("annotations.jsonl", dir_fd=fd)
    finally:
        os.close(fd)


def musdb_check(root, corpus, categories, out, also=None):
    """Step 2's access profile on a synthetic project: read the corpus manifest, the registry,
    the alias file and the titles, run the matcher and write the public report only. ``also``
    names one extra file to open (a negative control). This is test code: the real step-2
    workflow is not implemented."""
    root = Path(root)
    records = json.loads(Path(corpus).read_text(encoding="utf-8"))["recordings"]
    registry = json.loads((root / "manifests" / "split-registry.json").read_text(encoding="utf-8"))
    aliases = identity.read_aliases(root / "identity" / "aliases.json")
    titles = json.loads((root / "identity" / "titles.json").read_text(encoding="utf-8"))
    if also:
        with open(also, encoding="utf-8") as file:
            file.read(1)
    result = musdb.match(records, registry, aliases, titles, categories)
    hashes = {name: integrity.file_sha256(path) for name, path in (
        ("corpus_manifest", corpus), ("registry", root / "manifests" / "split-registry.json"),
        ("aliases", root / "identity" / "aliases.json"), ("titles", root / "identity" / "titles.json"))}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "public.json").write_text(json.dumps(musdb.public_report(result, hashes, "0" * 40)))


def raise_after_reading(root):
    """An error path: the target fails after the gateway has loaded its records."""
    records = gateway.Gateway(root, step=3).records()
    raise RuntimeError(json.dumps({"loaded": len(records)}))
