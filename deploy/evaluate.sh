#!/usr/bin/env bash
set -euo pipefail
: "${S4_AT_OUTPUT_ROOT:?}"
: "${S4_AT_ENV_ROOT:?}"
PY="$S4_AT_ENV_ROOT/bin/python"
test -x "$PY"
test -f "$S4_AT_OUTPUT_ROOT/accepted.json"
"$PY" - "$S4_AT_OUTPUT_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); a=json.loads((p/'accepted.json').read_text())
if a.get('state')!='ACCEPTED' or a.get('epochs')!=200 or a.get('primary')!='last.pt@200': raise SystemExit('acceptance identity mismatch')
m=p/'last_primary/metrics.json'
if not m.is_file(): raise SystemExit('last.pt@200 primary metrics missing')
print(json.dumps({'state':a['state'],'epochs':a['epochs'],'primary':a['primary'],'last_sha256':a.get('last_sha256'),'metrics':json.loads(m.read_text())},indent=2))
PY