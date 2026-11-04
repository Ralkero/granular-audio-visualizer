import importlib.util
from pathlib import Path
import pytest


def test_model_setup_refuses_unverified_bytes(tmp_path):
    path = Path(__file__).parents[1] / "experiments/separation/setup_model.py"
    spec = importlib.util.spec_from_file_location("model_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = tmp_path / "weights.safetensors"
    fake.write_bytes(b"not the pinned model")
    with pytest.raises(ValueError, match="checksum"):
        module.verify(fake)
    assert module.REVISION in module.URL and module.SIZE < 100 * 1024**2
