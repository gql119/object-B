#!/usr/bin/env bash
set -euo pipefail
echo "Refusing to signal a formal trainer: cloud cooperative-stop/recovery is not validated." >&2
echo "Use natural completion; retain all checkpoint and log files." >&2
exit 2
