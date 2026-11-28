#!/usr/bin/env bash
# The Phase A5.2 experiments environment: CPU-only PyTorch, separate from the application's
# environment and never imported by the application. Linux. Re-running it is harmless.
#
#   bash experiments/presence/setup_environment.sh [environment folder]
#
# Default folder: ~/.venvs/gmv-presence-experiments (outside the repository).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target="${1:-$HOME/.venvs/gmv-presence-experiments}"
python="$target/bin/python"

echo "== creating $target (Python 3.12)"
uv venv --python 3.12 --allow-existing "$target"
echo "== CPU-only PyTorch (no CUDA, no ROCm)"
uv pip install --python "$python" "torch==2.9.1+cpu" --index-url https://download.pytorch.org/whl/cpu
echo "== the rest, pinned by the lock file when it exists"
if [ -f "$here/experiments/presence/requirements-lock.txt" ]; then
  # torch is installed above from the CPU index; everything else comes from PyPI at the locked versions.
  grep -v -e '^torch==' -e '^-e ' -e '^granular-music-visualizer' "$here/experiments/presence/requirements-lock.txt" \
    | uv pip install --python "$python" -r -
else
  uv pip install --python "$python" numpy scipy soundfile pydantic psutil "mir_eval==0.8.2" pytest
fi
echo "== this checkout's gmv package, without its application dependencies"
uv pip install --python "$python" --no-deps -e "$here"
echo "== check"
"$python" - <<'EOF'
import platform, torch, numpy, scipy, soundfile
print("python", platform.python_version(), "| torch", torch.__version__, "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| soundfile", soundfile.__version__)
print("CUDA available:", torch.cuda.is_available(), "| ROCm build:", torch.version.hip is not None,
      "| CPU threads:", torch.get_num_threads())
import gmv.character.research.presence.loaders  # noqa: F401  (the protected loaders import cleanly)
print("gmv presence loaders: importable")
EOF
echo "== done"
