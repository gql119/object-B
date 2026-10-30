#!/usr/bin/env bash
set -euo pipefail
: "${S4_AT_SOURCE_ROOT:?set source root}"
: "${S4_AT_ENV_ROOT:?set data-volume Python environment}"
: "${S4_AT_CLEAN_ROOT:?set clean data root}"
: "${S4_AT_S4_PARENT_ROOT:?set accepted S4 parent root}"
: "${S4_AT_DATA_MANIFEST:?set full SHA manifest}"
: "${S4_AT_CPU_FIXTURE_ROOT:?set exact CPU fixture root}"
: "${S4_AT_CLOUD_GATE_ROOT:?set new cloud gate root}"
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
PY="$S4_AT_ENV_ROOT/bin/python"
test -x "$PY"
export PATH="$S4_AT_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$S4_AT_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test "$(git -C "$S4_AT_SOURCE_ROOT" rev-parse HEAD)" = "$S4_AT_EXPECTED_SOURCE_COMMIT"
"$PY" "$HERE/cloud_storage_check.py" --source-root "$S4_AT_SOURCE_ROOT" --phase run
test -f "$S4_AT_OWNED_ROOT/config/path_aliases.json"
"$PY" "$HERE/data/verify_data_manifest.py" --manifest "$S4_AT_DATA_MANIFEST" --clean-root "$S4_AT_CLEAN_ROOT" --s4-parent-root "$S4_AT_S4_PARENT_ROOT"
if ! test -d "$S4_AT_CPU_FIXTURE_ROOT"; then
  "$PY" "$HERE/data/make_cpu_fixtures.py" --clean-root "$S4_AT_CLEAN_ROOT" --out "$S4_AT_CPU_FIXTURE_ROOT"
fi
"$PY" "$S4_AT_SOURCE_ROOT/robustness_s4_at/validate_cpu.py" --fixtures "$S4_AT_CPU_FIXTURE_ROOT" --out "$S4_AT_CLOUD_GATE_ROOT/cpu"
"$PY" - "$S4_AT_SOURCE_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); p=json.loads((root/'robustness_s4_at/configs/PROTOCOL.json').read_text())
from robustness_s4_at.common import protocol, native_identity, scientific_source
protocol(); native_identity(); closure=scientific_source()
if closure['uncommitted_files']: raise SystemExit('uncommitted source files')
r=__import__('robustness_s4_at.queue',fromlist=['resource_status']).resource_status()
print(json.dumps(r,indent=2))
if not r['admitted']: raise SystemExit('resource gate blocked')
PY
"$PY" "$S4_AT_SOURCE_ROOT/robustness_s4_at/validate_cuda.py" --out "$S4_AT_CLOUD_GATE_ROOT/cuda" --source-commit "$S4_AT_EXPECTED_SOURCE_COMMIT"
echo "Evidence collected only. This script does not create accepted preflight or scientific approval."