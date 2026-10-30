#!/usr/bin/env bash
set -euo pipefail
: "${S4_AT_SOURCE_ROOT:?set source root}"
: "${S4_AT_OWNED_ROOT:?set owned output root}"
: "${S4_AT_STORAGE_ROOT:?set mounted data-volume root}"
: "${S4_AT_ENV_ROOT:?set environment path on data volume}"
: "${S4_AT_STORAGE_PROFILE:?set cloud storage profile}"
test "$S4_AT_STORAGE_PROFILE" = autodl_30g50g_v1 || { echo "Unknown storage profile" >&2; exit 2; }
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "$S4_AT_STORAGE_ROOT" "$S4_AT_OWNED_ROOT" "$S4_AT_ENV_ROOT" "$TMPDIR" \
  "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
python3 "$HERE/cloud_storage_check.py" --source-root "$S4_AT_SOURCE_ROOT" --phase install
test "${S4_AT_INSTALL_DIRECT_PINS:-NO}" = YES || { echo "Direct pins are incomplete; set S4_AT_INSTALL_DIRECT_PINS=YES only for non-formal environment preparation." >&2; exit 2; }
if test -e "$S4_AT_ENV_ROOT/bin/python"; then
  echo "Environment already exists; refusing to overwrite it" >&2
  exit 2
fi
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
test -x "$CONDA_BIN" || { echo "BLOCKED: conda executable not found; provision Python 3.11 in the data volume first" >&2; exit 2; }
"$CONDA_BIN" create -y -p "$S4_AT_ENV_ROOT" python=3.11 pip
PY="$S4_AT_ENV_ROOT/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version; print(sys.version)'
export PIP_NO_CACHE_DIR=1
"$PY" -m pip install --requirement "$HERE/requirements.lock"
export S4_AT_PACKAGE_ROOT="$S4_AT_SOURCE_ROOT"
"$PY" "$HERE/install_native_sources.py"
"$PY" -m pip check
"$PY" - <<'PY'
import json,sys,torch,torchvision,ultralytics
assert sys.version_info[:2] == (3,11), sys.version
assert torch.__version__.startswith('2.7.1+cu128'), torch.__version__
assert torchvision.__version__.startswith('0.22.1+cu128'), torchvision.__version__
assert ultralytics.__version__ == '8.4.150', ultralytics.__version__
assert torch.cuda.is_available(), 'GPU mode is not visible to PyTorch'
free,total=torch.cuda.mem_get_info(0)
print(json.dumps({'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
 'ultralytics':ultralytics.__version__,'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0),
 'free_MiB':free/2**20,'total_MiB':total/2**20},indent=2))
PY
OBSERVED="$S4_AT_OWNED_ROOT/environment/requirements_observed.txt"
mkdir -p "$(dirname "$OBSERVED")"
test ! -e "$OBSERVED" || { echo "Observed environment manifest already exists; refusing overwrite" >&2; exit 2; }
"$PY" -m pip freeze > "$OBSERVED"
echo "Environment prepared. Direct pins plus observed transitive package versions are retained; scientific CUDA/preflight gates remain pending."