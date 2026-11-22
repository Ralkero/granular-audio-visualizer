"""Generate docs/schemas/*.json from the pydantic models so they cannot go stale.

Run ``python -m gmv.schema_export`` after changing a model; ``--check`` fails when the
committed files differ (the test suite runs the same comparison).
"""

import argparse
import json
from pathlib import Path

from . import models

ROOT = Path(__file__).resolve().parents[2] / "docs" / "schemas"


def schema_models():
    result = {
        name: getattr(models, name)
        for name in (
            "AnalysisConfig",
            "FeatureFrame",
            "MusicalEvent",
            "Status",
            "InstrumentEstimate",
            "InstrumentActivity",
        )
    }
    try:
        from .corpus import schema
    except ImportError:
        return result
    result.update(CorpusManifest=schema.CorpusManifest, AnnotationFile=schema.AnnotationFile)
    return result


def render(model):
    return json.dumps(model.model_json_schema(), indent=2) + "\n"


def stale(root=ROOT):
    return [
        name
        for name, model in schema_models().items()
        if not (root / f"{name}.json").is_file()
        or (root / f"{name}.json").read_text(encoding="utf-8") != render(model)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        outdated = stale()
        if outdated:
            raise SystemExit(f"Stale schema files: {outdated}; run python -m gmv.schema_export")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, model in schema_models().items():
        (ROOT / f"{name}.json").write_text(render(model), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
