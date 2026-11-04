"""Explicit 1.0 -> 1.1 migration and conservative live-impact policy."""

from . import models, models_v1


def parse_message(data):
    version = data.get("schema_version", "1.0")
    classes = {"feature_frame": "FeatureFrame", "musical_event": "MusicalEvent", "status": "Status"}
    name = classes[data["type"]]
    if version == "1.0":
        legacy = getattr(models_v1, name).model_validate(data)
        data = legacy.model_dump()
        data["schema_version"] = "1.1"
        if name == "MusicalEvent":
            data.update(provenance="legacy_unknown", intended_use="retrospective")
        if name == "FeatureFrame":
            data["measurement_semantics"] = "raw_channel_mean_v1; no_layout_analysis"
    elif version != "1.1":
        raise ValueError(f"Unsupported schema version: {version}")
    return getattr(models, name).model_validate(data)


def eligible_for_live_impact(message, *, now, session_id, segment, max_age_ms=100):
    """Unknown clocks, replay, predictions, proxies and stale results fail closed."""
    return bool(
        isinstance(message, models.MusicalEvent)
        and not message.replayed
        and message.session_id == session_id
        and message.segment == segment
        and message.analysis_tier == "realtime"
        and message.intended_use == "immediate"
        and message.evidence_kind in ("observed", "candidate")
        and message.evidence_monotonic is not None
        and 0 <= (now - message.evidence_monotonic) * 1000 <= max_age_ms
    )
