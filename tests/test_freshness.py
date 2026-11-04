from threading import Event
import json
from pathlib import Path
import pytest
from gmv.bus import EventBus
from gmv.contract import eligible_for_live_impact, parse_message
from gmv.models import MusicalEvent
from gmv.session import replay


def event(**updates):
    values = dict(
        session_id="s",
        source="test",
        timestamp=1,
        detected_monotonic=10.01,
        evidence_monotonic=10,
        segment=2,
        event_id="e",
        event_type="onset",
        intensity=0.5,
        confidence=0.5,
        method="test",
        evidence_kind="observed",
    )
    values.update(updates)
    return MusicalEvent(**values)


def test_freshness_rejects_delayed_replay_prediction_and_other_segment():
    options = dict(now=10.05, session_id="s", segment=2)
    assert eligible_for_live_impact(event(), **options)
    for updates in (
        {"evidence_monotonic": 7},
        {"replayed": True},
        {"segment": 3},
        {"session_id": "other"},
        {"analysis_tier": "advanced"},
        {"evidence_kind": "predicted"},
        {"evidence_monotonic": None},
        {"evidence_monotonic": 11},
        {"intended_use": "context"},
    ):
        assert not eligible_for_live_impact(event(**updates), **options)


def test_legacy_fixture_replay_preserves_evidence_and_original_publication():
    path = Path(__file__).parent / "fixtures/schema_1_0.jsonl"
    originals = [json.loads(line) for line in path.read_text().splitlines()]
    bus = EventBus()
    queue = bus.subscribe()
    replay(path, bus, Event(), speed=1000)
    for old in originals:
        new = queue.get(timeout=0)
        assert new.schema_version == "1.1" and new.replayed
        assert new.timestamp == old["timestamp"]
        assert new.original_published_monotonic == old["published_monotonic"]
        assert new.age_at_publication_ms is None
    bad = dict(originals[0], schema_version="2.0")
    with pytest.raises(ValueError):
        parse_message(bad)
    with pytest.raises(ValueError):
        parse_message(dict(originals[0], unexpected=True))


def test_signed_age_is_not_clamped():
    bus = EventBus()
    result = bus.publish(event(evidence_monotonic=1e12))
    assert result.age_at_publication_ms < 0
