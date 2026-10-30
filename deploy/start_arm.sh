#!/usr/bin/env bash
set -euo pipefail
: "${S4_AT_SOURCE_ROOT:?}"
: "${S4_AT_ENV_ROOT:?}"
: "${S4_AT_OUTPUT_ROOT:?}"
: "${S4_AT_ACCEPTED_PREFLIGHT:?}"
: "${S4_AT_RUN_ID:?}"
: "${S4_AT_EXPECTED_SOURCE_COMMIT:?}"
test -d "$S4_AT_SOURCE_ROOT/.git"
test -x "$S4_AT_ENV_ROOT/bin/python" || { echo "data-volume Python environment missing; run deploy/setup_env.sh first" >&2; exit 2; }
export PATH="$S4_AT_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$S4_AT_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -f "$S4_AT_ACCEPTED_PREFLIGHT"
test ! -e "$S4_AT_OUTPUT_ROOT" || { echo "Output exists; refusing restart or overwrite" >&2; exit 2; }
source_sha="$(git -C "$S4_AT_SOURCE_ROOT" rev-parse HEAD)"
test "$source_sha" = "$S4_AT_EXPECTED_SOURCE_COMMIT" || { echo "Release SHA mismatch" >&2; exit 2; }
python "$(dirname "$0")/cloud_storage_check.py" --source-root "$S4_AT_SOURCE_ROOT" --phase run
control="$S4_AT_OWNED_ROOT/control"; mkdir -p "$control"
pidfile="$control/${S4_AT_RUN_ID}.pid"; logfile="$control/${S4_AT_RUN_ID}.log"
test ! -e "$pidfile" && test ! -e "$logfile" || { echo "Control artifact collision; retain and choose a fresh RunID" >&2; exit 2; }
nohup env PYTHONPATH="$S4_AT_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$S4_AT_ENV_ROOT/bin/python" -B -m robustness_s4_at.worker \
  --formal --run-id "$S4_AT_RUN_ID" --out "$S4_AT_OUTPUT_ROOT" --source-commit "$source_sha" \
  --preflight "$S4_AT_ACCEPTED_PREFLIGHT" --device 0 >"$logfile" 2>&1 < /dev/null &
pid=$!; printf '%s\n' "$pid" > "$pidfile"
printf 'started PID=%s\nlog=%s\n' "$pid" "$logfile"