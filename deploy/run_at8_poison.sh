#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$HERE/../cloud_configs/S4_CLOUD_COMMON.env.example"
source "$HERE/../cloud_configs/S4_CLOUD_POISON_AT8.env.example"
export S4_AT_SOURCE_COMMIT="$(git -C "$S4_AT_SOURCE_ROOT" rev-parse HEAD)"
export S4_AT_OUTPUT_ROOT="$S4_AT_OWNED_ROOT/outputs/$S4_AT_RUN_ID"
exec bash "$HERE/start_arm.sh"
