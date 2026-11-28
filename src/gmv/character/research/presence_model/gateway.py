"""The only route from the protected presence loader to experiment code (readiness proposal,
revision 4.4, section 6).

* It calls ``loaders.development_examples(root, ("development",))`` and nothing else, so the
  loader's integrity checks run unchanged.
* It passes on only the Tier 2 fields and the folded annotation state. ``source`` is never
  passed to any module. The raw ``selection`` string never leaves the gateway.
* In step 6 only, and only once a seal record exists, it releases one category token per
  excerpt, parsed against a vocabulary. The production vocabulary is **not decided**: a
  vocabulary must be marked approved to be used, unless a caller (a synthetic test) says
  otherwise explicitly. ``tracked_code_candidate`` derives a candidate from the tracked
  selection code; it is not approved.

The parser uses ``re.fullmatch`` without flags (case-sensitive; a trailing newline or any
whitespace does not match), accepts only ``str`` values, never captures reason text, never
raises on content and never logs. Tokens come from a table built from the vocabulary, never
from input characters. Fallbacks: ``missing`` and ``unrecognised``.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..presence import loaders

FIELDS = ("id", "recording", "path", "size", "sample_rate", "channels", "duration_s", "start_s",
          "end_s", "group", "status", "confirmed", "uncertain")
MISSING, UNRECOGNISED = "missing", "unrecognised"
F1 = r"phase A3 development excerpt \(delivery answer: ({})\)"
F2 = r"negative candidate, ({}): .+"


class GatewayError(PermissionError):
    """A request outside the gateway's rules (messages are fixed texts)."""


@dataclass(frozen=True)
class Vocabulary:
    a3: tuple
    candidate: tuple
    approved: bool = False


def tracked_code_candidate():
    """The values the tracked selection code can generate: a candidate, not approved."""
    from ..presence import selection

    return Vocabulary(tuple(name for name, _quota in selection.QUOTAS),
                      (*selection.NEGATIVE_RULES, selection.INTRO), approved=False)


def _pattern(template, values):
    return re.compile(template.format("|".join(re.escape(v) for v in values)))


def parse(value, vocabulary):
    """One fixed token for a raw ``selection`` value."""
    if value is None or value == "":
        return MISSING
    if type(value) is not str:
        return UNRECOGNISED
    a3 = {v: f"A3:{v}" for v in vocabulary.a3}
    candidate = {v: f"candidate:{v}" for v in vocabulary.candidate}
    for template, table in ((F1, a3), (F2, candidate)):
        if table:
            match = _pattern(template, table).fullmatch(value)
            if match is not None:
                return table[match.group(1)]
    return UNRECOGNISED


class Gateway:
    def __init__(self, root, step):
        records = loaders.development_examples(Path(root), ("development",))
        self.step = step
        self._records = [{k: r[k] for k in FIELDS} for r in records]
        self._selection = {r["id"]: r.get("selection") for r in records} if step == 6 else None

    def records(self):
        return [dict(r) for r in self._records]

    def release_tokens(self, seal_path, vocabulary, require_approved=True):
        if self._selection is None:
            raise GatewayError("category tokens are released only in step 6")
        if require_approved and not vocabulary.approved:
            raise GatewayError("the selection vocabulary has not been approved")
        try:
            seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise GatewayError("no seal record: tokens are released only after sealing") from None
        if seal.get("sealed") is not True or not seal.get("predictions_sha256"):
            raise GatewayError("the seal record is incomplete")
        return {excerpt: parse(value, vocabulary) for excerpt, value in self._selection.items()}


def breakdown(tokens, group_of, metric_of=None, minimum_groups=2):
    """Descriptive counts per token. Categories with fewer than ``minimum_groups`` groups are
    suppressed (counted, never listed); fallback buckets show counts only, never metrics."""
    rows, suppressed = {}, []
    for token in sorted(set(tokens.values())):
        excerpts = [e for e, t in tokens.items() if t == token]
        groups = {group_of[e] for e in excerpts}
        if len(groups) < minimum_groups:
            suppressed.append(len(excerpts))
            continue
        row = {"groups": len(groups), "excerpts": len(excerpts)}
        if metric_of is not None and token not in (MISSING, UNRECOGNISED):
            row["metric"] = sum(metric_of[e] for e in excerpts) / len(excerpts)
        rows[token] = row
    summary = {"suppressed_categories": len(suppressed)}
    if len(suppressed) > 1:
        summary["suppressed_excerpts"] = sum(suppressed)
    return {"categories": rows, "suppressed": summary}
