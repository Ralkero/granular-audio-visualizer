"""Protected reading of the presence project.

* ``check_project`` verifies every manifest and the registry against their fingerprints,
  and every excerpt against the registry: its group exists, is in the excerpt's partition,
  and holds the excerpt's recording; no excerpt or recording appears in two partitions;
  no reserved or excluded group appears in a manifest; and the registry's frozen final-test
  cohort agrees with the final-test manifest (``project.check_final_test``), so an
  interrupted freeze is refused. Manifests hold no annotations, so reading the final-test
  manifest here exposes no answer.
* ``development_examples`` is the only loader for development work. It reads the
  development and validation annotation histories and nothing else: asking it for the final
  test raises ``ProtectedSplitError``, and it never opens a path in the final-test folder.
  A broken history (chain) is refused.

The final-test annotations are written and read only by the annotation window; this module
offers no way to read them, and Phase A5.1 implements no final scoring.
"""

from . import project, schema
from .store import Store

DEVELOPMENT_PARTITIONS = ("development", "validation")


class ProtectedSplitError(PermissionError):
    """A development loader was asked for final-test material."""


def check_project(root):
    """(registry, manifests by partition) after every integrity check; raises
    ``project.IntegrityError`` listing the kinds of problem found."""
    registry = project.read_registry(root)
    entries = registry.get("entries", {})
    manifests = {p: project.read_manifest(root, p) for p in project.MANIFESTS}
    problems = set()
    owner_of_id, owner_of_recording = {}, {}
    for partition, manifest in manifests.items():
        for excerpt in (manifest or {}).get("excerpts", []):
            entry = entries.get(excerpt["group"])
            if entry is None:
                problems.add("an excerpt's artist group is not in the registry")
            else:
                if entry["partition"] != partition:
                    problems.add(f"an excerpt's artist group belongs to {entry['partition']}, not {partition}")
                if f"rec:{excerpt['recording']}" not in entry["nodes"]:
                    problems.add("an excerpt's recording is not a member of its artist group")
            if owner_of_id.setdefault(excerpt["id"], partition) != partition:
                problems.add("an excerpt appears in two partitions")
            if owner_of_recording.setdefault(excerpt["recording"], partition) != partition:
                problems.add("a recording appears in two partitions")
    problems.update(project.check_final_test(registry, manifests.get("final_test")))
    if problems:
        raise project.IntegrityError("; ".join(sorted(problems)))
    return registry, manifests


def development_examples(root, partitions=DEVELOPMENT_PARTITIONS):
    """Every excerpt of the requested development partitions with its current annotation:
    ``status`` (``schema.STATUSES``), confirmed and uncertain spans (unions), tags. Only
    ``development`` and ``validation`` may be requested."""
    partitions = tuple(partitions)
    if not partitions or any(p not in DEVELOPMENT_PARTITIONS for p in partitions):
        raise ProtectedSplitError("development loaders read only the development and validation partitions")
    _registry, manifests = check_project(root)
    out = []
    for partition in partitions:
        manifest = manifests.get(partition)
        if not manifest:
            continue
        history = Store(project.partition_folder(root, partition), project.durations(manifest))
        if not history.intact:
            raise project.IntegrityError(f"the {partition} annotation history is broken at event {history.broken_at}")
        states = history.state()
        for excerpt in manifest["excerpts"]:
            state = states.get(excerpt["id"])
            out.append({
                **excerpt,
                "partition": partition,
                "status": schema.status(state),
                "confirmed": schema.spans(state, "confirmed") if state else [],
                "uncertain": schema.spans(state, "uncertain") if state else [],
                "tags": list(state["tags"]) if state else [],
            })
    return out
