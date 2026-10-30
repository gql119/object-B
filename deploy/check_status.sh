#!/usr/bin/env bash
set -euo pipefail
: "${S4_AT_OUTPUT_ROOT:?}"
if test -n "${S4_AT_OWNED_ROOT:-}" && test -n "${S4_AT_RUN_ID:-}"; then
  pidfile="$S4_AT_OWNED_ROOT/control/${S4_AT_RUN_ID}.pid"
  logfile="$S4_AT_OWNED_ROOT/control/${S4_AT_RUN_ID}.log"
  if test -f "$pidfile"; then pid="$(cat "$pidfile")"; if kill -0 "$pid" 2>/dev/null; then echo "PID $pid appears live"; else echo "PID $pid is not live"; fi; fi
  if test -f "$logfile"; then tail -n 40 "$logfile"; fi
fi
if test -f "$S4_AT_OUTPUT_ROOT/accepted.json"; then cat "$S4_AT_OUTPUT_ROOT/accepted.json"; exit 0; fi
if test -f "$S4_AT_OUTPUT_ROOT/training/results.csv"; then tail -n 3 "$S4_AT_OUTPUT_ROOT/training/results.csv"; else echo "No results.csv yet"; fi
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader; fi
