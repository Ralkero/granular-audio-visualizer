from gmv import schema_export


def test_committed_json_schemas_match_models():
    assert schema_export.stale() == []


def test_stale_detects_missing_and_changed_files(tmp_path):
    names = set(schema_export.schema_models())
    assert set(schema_export.stale(tmp_path)) == names
    for name, model in schema_export.schema_models().items():
        (tmp_path / f"{name}.json").write_text(schema_export.render(model), encoding="utf-8")
    assert schema_export.stale(tmp_path) == []
    (tmp_path / "Status.json").write_text("{}", encoding="utf-8")
    assert schema_export.stale(tmp_path) == ["Status"]
