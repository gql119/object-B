#!/usr/bin/env python3
"""Print and enforce one cloud storage phase budget without summing mounts."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--phase', choices=('install', 'data_stage', 'run', 'monitor'), required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root.resolve()))
from robustness_s4_at.cloud_storage import storage_status
result = storage_status(args.phase)
print(json.dumps(result, indent=2, sort_keys=True))
if not result['admitted']:
    raise SystemExit(2)
