"""The A5.2 seed plan: every production seed identity, the random streams each consumer
receives, and the checks that must pass before any training (readiness proposal, revision
4.4, section 11). NumPy only.

**Provisional.** The string formats are revision 4.4's proposed fill-ins for the registered
template ``gmv-a5.2:{arm}:{outer}:{inner}:{seed_index}`` (protocol §7). They are not
registered until Amendment 3. The registered formula itself (``seeds.training_seed``) is
not changed here: ``integer_of`` applies it to whole strings.

Families (policy C; policy D keeps the leading seed indices):

* registered inner ``gmv-a5.2:{arm}:{o}:{i}:{s}``, o 0-18, i 0-4, s 0-1 (D: 0);
* registered outer ``gmv-a5.2:{arm}:{o}:outer:{s}``, s 0-4 (D: 0-2);
* registered final ``gmv-a5.2:{arm}:final:final:{s}``, s 0-4 (D: 0-2);
* gate ``gmv-a5.2-gate:{arm}:{set}:{purpose}:{g}``, the fixed list ``GATE_RUNS``;
* sensitivity inner and outer ``gmv-a5.2-{ns}:{arm}:{o}:...``, ns sensT or sensV, o 0-17.

**Consumers.** Each identity's integer n seeds one ``SeedSequence(n, spawn_key=key)`` per
consumer. NumPy consumers build ``PCG64`` from it directly; PyTorch consumers are seeded with
its first 64-bit state word. The uniqueness checks compare exactly what each kind of consumer
receives. Distinct values establish non-collision at that level only; they do not show that
random sequences are disjoint, nor that training runs are statistically independent.

**Collisions.** Registered identities are never remapped: any collision stops the plan
(``SeedCollisionError``). Gate and sensitivity identities, processed in sorted Unicode order,
take the suffix ``:c1``, ``:c2``, ... until their integer and every consumer value are unique.

**Replays.** ``RunLedger`` accepts a seed string a second time only for a declared replay
(``REPLAYS``) whose source has already run, and each declared replay once; any other reuse
stops.

**Configuration.** ``configuration`` gives the seed format version and the plan's enumeration
fingerprint, which every integrity record carries in R-E.
"""

import functools
import hashlib
from dataclasses import dataclass

import numpy as np

from . import seeds

FORMAT = "a5.2-seed-format-v1"
ARMS = ("B1", "C")
OUTER_FOLDS = 19
INNER_FOLDS = 5
SENSITIVITY_OUTER_FOLDS = 18  # F0 removes at least one group
NAMESPACES = ("sensT", "sensV")
SEED_INDICES = {
    "C": {"inner": (0, 1), "outer": (0, 1, 2, 3, 4), "final": (0, 1, 2, 3, 4)},
    "D": {"inner": (0,), "outer": (0, 1, 2), "final": (0, 1, 2)},
}
GATE_RUNS = (("outer-0", "conv", 0), ("outer-0", "conv", 1), ("inner-0", "time", 0),
             ("inner-0", "conv", 0), ("inner-0", "conv", 1),
             ("outer-0", "par", 0), ("outer-0", "par", 1), ("outer-0", "par", 2), ("outer-0", "par", 3))
GATE_FALLBACK = {"C": "conv45", "B1": "conv90"}
CONSUMERS = {
    "C": (("init", "torch", (0,)), ("order", "numpy", (1,)),
          ("augment-mix", "numpy", (2, 0)), ("augment-gain", "numpy", (2, 1)),
          ("augment-tilt", "numpy", (2, 2)), ("augment-mask", "numpy", (2, 3)),
          ("dropout", "torch", (3,))),
    "B1": (("init", "numpy", (0,)), ("order", "numpy", (1,))),
}
REPLAYS = {"G-C3": "gmv-a5.2-gate:C:outer-0:conv:0", "G-B1-3": "gmv-a5.2-gate:B1:outer-0:conv:0"}
MAX_SUFFIX = 1000


class SeedPlanError(RuntimeError):
    """The seed plan is malformed or a run asked for a seed it may not use."""


class SeedCollisionError(SeedPlanError):
    """Two seed identities would receive the same initialisation; the run stops."""


class ReplayError(SeedPlanError):
    """A replay is undeclared, mismatched or not deterministic."""


# -- enumeration ---------------------------------------------------------------------------------


def registered(policy="C"):
    s = SEED_INDICES[policy]
    out = []
    for arm in ARMS:
        out += [f"gmv-a5.2:{arm}:{o}:{i}:{k}" for o in range(OUTER_FOLDS) for i in range(INNER_FOLDS) for k in s["inner"]]
        out += [f"gmv-a5.2:{arm}:{o}:outer:{k}" for o in range(OUTER_FOLDS) for k in s["outer"]]
        out += [f"gmv-a5.2:{arm}:final:final:{k}" for k in s["final"]]
    return out


def gate():
    out = []
    for arm in ARMS:
        runs = [*GATE_RUNS, ("outer-0", GATE_FALLBACK[arm], 0), ("outer-0", GATE_FALLBACK[arm], 1)]
        out += [f"gmv-a5.2-gate:{arm}:{part}:{purpose}:{g}" for part, purpose, g in runs]
    return out


def sensitivity(policy="C"):
    s = SEED_INDICES[policy]
    out = []
    for ns in NAMESPACES:
        for arm in ARMS:
            folds = range(SENSITIVITY_OUTER_FOLDS)
            out += [f"gmv-a5.2-{ns}:{arm}:{o}:{i}:{k}" for o in folds for i in range(INNER_FOLDS) for k in s["inner"]]
            out += [f"gmv-a5.2-{ns}:{arm}:{o}:outer:{k}" for o in folds for k in s["outer"]]
    return out


def production(policy="C"):
    """Every production seed identity for ``policy`` (gate identities do not depend on it)."""
    return registered(policy) + gate() + sensitivity(policy)


def family_of(text):
    if text.startswith("gmv-a5.2:"):
        return "registered"
    if text.startswith("gmv-a5.2-gate:"):
        return "gate"
    if text.startswith(tuple(f"gmv-a5.2-{ns}:" for ns in NAMESPACES)):
        return "sensitivity"
    raise SeedPlanError("not an A5.2 seed identity")


def arm_of(text):
    arm = text.split(":")[1]
    if arm not in ARMS:
        raise SeedPlanError("unknown arm in a seed identity")
    return arm


# -- derivation ----------------------------------------------------------------------------------


def integer_of(text):
    """The registered derivation applied to a whole string: the first 8 hex digits of its
    SHA-256, as an integer (equal to ``seeds.training_seed`` for registered strings)."""
    return int(seeds.sha256_hex(text)[:8], 16)


def seed_sequence(n, key):
    return np.random.SeedSequence(n, spawn_key=key)


def torch_seed_of(n, key):
    """The 64-bit value a PyTorch consumer is seeded with."""
    return int(seed_sequence(n, key).generate_state(1, np.uint64)[0])


def numpy_state_of(n, key):
    """The initial (state, increment) of the PCG64 generator a NumPy consumer receives."""
    state = np.random.PCG64(seed_sequence(n, key)).state["state"]
    return int(state["state"]), int(state["inc"])


def numpy_generator(n, key):
    return np.random.Generator(np.random.PCG64(seed_sequence(n, key)))


def consumer_key(arm, consumer):
    for name, _kind, key in CONSUMERS[arm]:
        if name == consumer:
            return key
    raise SeedPlanError("unknown consumer")


def canonical_bytes(lines):
    return "\n".join(sorted(lines)).encode("utf-8")


def enumeration_sha256(strings, integer_of=integer_of):
    """Canonical serialisation v1: ``f"{string}\\t{integer}"`` per identity, sorted in Python
    string order, joined with newlines, no trailing newline, UTF-8, SHA-256 (lower-case hex)."""
    return hashlib.sha256(canonical_bytes(f"{s}\t{integer_of(s)}" for s in strings)).hexdigest()


# -- the plan ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedPlan:
    format: str
    policy: str
    identities: tuple  # (base string, final string, integer), sorted by base string
    sha256: str  # enumeration_sha256 over the final strings
    consumers_sha256: str
    remapped: int
    torch_values: int
    numpy_states: int

    def final(self, base):
        for b, f, _n in self.identities:
            if b == base:
                return f
        raise SeedPlanError("not in the seed plan")


def plan(policy="C", strings=None, integer_of=integer_of, torch_seed_of=torch_seed_of,
         numpy_state_of=numpy_state_of):
    """Check every identity and return the plan; the derivation functions are injectable so
    synthetic tests can force collisions. Raises ``SeedCollisionError`` on any registered
    collision; remaps colliding gate and sensitivity identities."""
    strings = production(policy) if strings is None else list(strings)
    if len(set(strings)) != len(strings):
        raise SeedPlanError("the seed plan lists a string twice")
    used_integers, used_torch, used_numpy = set(), set(), set()
    consumer_lines, identities = [], []

    def values(text, n):
        torch_values, numpy_values, lines = [], [], []
        for name, kind, key in CONSUMERS[arm_of(text)]:
            if kind == "torch":
                value = torch_seed_of(n, key)
                torch_values.append(value)
                lines.append(f"{text}\t{name}\ttorch\t{value}")
            else:
                value = numpy_state_of(n, key)
                numpy_values.append(value)
                lines.append(f"{text}\t{name}\tnumpy\t{value[0]}:{value[1]}")
        return torch_values, numpy_values, lines

    def free(n, torch_values, numpy_values):
        return (n not in used_integers and len(set(torch_values)) == len(torch_values)
                and len(set(numpy_values)) == len(numpy_values)
                and not used_torch.intersection(torch_values) and not used_numpy.intersection(numpy_values))

    def accept(base, final, n, torch_values, numpy_values, lines):
        used_integers.add(n)
        used_torch.update(torch_values)
        used_numpy.update(numpy_values)
        consumer_lines.extend(lines)
        identities.append((base, final, n))

    remapped = 0
    for text in sorted(s for s in strings if family_of(s) == "registered"):
        n = integer_of(text)
        found = values(text, n)
        if n in used_integers:
            raise SeedCollisionError("two registered seed identities share an integer seed")
        if not free(n, found[0], found[1]):
            raise SeedCollisionError("two registered consumers would share a derived stream seed")
        accept(text, text, n, *found)
    for base in sorted(s for s in strings if family_of(s) != "registered"):
        for suffix in range(MAX_SUFFIX + 1):
            candidate = base if suffix == 0 else f"{base}:c{suffix}"
            n = integer_of(candidate)
            found = values(candidate, n)
            if free(n, found[0], found[1]):
                remapped += suffix > 0
                accept(base, candidate, n, *found)
                break
        else:
            raise SeedCollisionError("no free seed after the suffix limit")
    identities.sort()
    finals = [f for _b, f, _n in identities]
    integers = {f: n for _b, f, n in identities}
    return SeedPlan(FORMAT, policy, tuple(identities),
                    enumeration_sha256(finals, integers.__getitem__),
                    hashlib.sha256(canonical_bytes(consumer_lines)).hexdigest(),
                    remapped, len(used_torch), len(used_numpy))


# -- runs and replays ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _checked_plan_sha256(policy):
    return plan(policy).sha256


def configuration(policy="C"):
    """The seed fields of the configuration record (R-E): the format version and the
    enumeration fingerprint of the checked plan (``plan`` stops on any collision)."""
    return {"seed_format": FORMAT, "seed_enumeration_sha256": _checked_plan_sha256(policy)}


HORIZON_SOURCES = ("outer-0:conv:0", "outer-0:conv:1", "inner-0:conv:0", "inner-0:conv:1")
LENGTHS = {"C": (30, 45, 68), "B1": (60, 90, 135)}  # E, E⁺ = 1.5E, E⁺⁺ = ⌈2.25E⌉ (A3.2)


def horizon_relations():
    """The 16 declared ``horizon`` relations (Amendment 3, A3.3): {relation id: (base gate
    string, shorter length, longer length)}, 2 arms × 4 gate sources × (E → E⁺, E⁺ → E⁺⁺)."""
    out = {}
    for arm in ARMS:
        e, e1, e2 = LENGTHS[arm]
        for source in HORIZON_SOURCES:
            base = f"gmv-a5.2-gate:{arm}:{source}"
            out[f"H-{arm}-{source}-{e1}"] = (base, e, e1)
            out[f"H-{arm}-{source}-{e2}"] = (base, e1, e2)
    return out


class RunLedger:
    """Which run used which seed identity. A production identity runs once; a declared replay
    runs once, reusing its source's string, only after the source has run. A declared
    ``horizon`` relation (A3.3) reruns a gate identity as a separate, longer complete recipe:
    its source must already have run at the shorter length (E, or the E⁺ relation for E⁺⁺),
    it is claimed once, and only gate identities have relations. Any other reuse stops."""

    def __init__(self, seed_plan, replays=REPLAYS, horizons=None):
        self.allowed = {f for _b, f, _n in seed_plan.identities}
        final = {b: f for b, f, _n in seed_plan.identities}
        self.replays = dict(replays)
        relations = horizon_relations() if horizons is None else horizons
        self.horizons = {k: (final.get(base, base), short, long) for k, (base, short, long) in relations.items()}
        self.used = {}
        self.lengths = {}
        self.replayed = set()
        self.claimed = set()

    def claim_production(self, run_id, text, epochs=None):
        if text not in self.allowed:
            raise SeedPlanError("a run asked for a seed identity outside the plan")
        if text in self.used:
            raise ReplayError("a production seed identity was reused without a declared replay")
        self.used[text] = run_id
        self.lengths[text] = epochs

    def claim_horizon(self, relation_id, text, epochs):
        if relation_id not in self.horizons:
            raise ReplayError("an undeclared horizon relation")
        source, short, long = self.horizons[relation_id]
        if text != source or family_of(text) != "gate":
            raise ReplayError("a horizon run's seed differs from its declared gate source")
        if epochs != long:
            raise ReplayError("a horizon run's length is not its relation's longer length")
        if relation_id in self.claimed:
            raise ReplayError("a horizon relation was claimed twice")
        if text not in self.used:
            raise ReplayError("a horizon run ran before its source")
        arm = arm_of(text)
        if short == LENGTHS[arm][0]:
            if self.lengths.get(text) != short:
                raise ReplayError("a horizon run's source has not run at the registered length")
        else:
            earlier = [k for k, (s, _a, b) in self.horizons.items() if s == text and b == short]
            if not earlier or not set(earlier) <= self.claimed:
                raise ReplayError("the E⁺⁺ run came before its E⁺ run")
        self.claimed.add(relation_id)

    def claim_replay(self, replay_id, text):
        if replay_id not in self.replays:
            raise ReplayError("an undeclared replay")
        if text != self.replays[replay_id]:
            raise ReplayError("a replay's seed differs from its declared source")
        if text not in self.used:
            raise ReplayError("a replay ran before its source")
        if replay_id in self.replayed:
            raise ReplayError("a declared replay was claimed twice")
        self.replayed.add(replay_id)


def verify_replay(source, replay):
    """Stop unless the replay reproduced its source exactly (bytes compared)."""
    if bytes(source) != bytes(replay):
        raise ReplayError("a determinism replay did not reproduce its source")
