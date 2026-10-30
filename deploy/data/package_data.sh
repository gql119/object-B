#!/usr/bin/env bash
set -euo pipefail
: "${S4_AT_CLEAN_ROOT:?set clean root}"
: "${S4_AT_S4_PARENT_ROOT:?set accepted S4 parent root}"
: "${S4_AT_DATA_MANIFEST:?set completed full source hash manifest}"
: "${S4_AT_EXPORT_ROOT:?set a new export directory}"
test -f "$S4_AT_DATA_MANIFEST"
python3 - "$S4_AT_DATA_MANIFEST" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
if m.get('state')!='FULL_SOURCE_HASH_SCAN_PASS': raise SystemExit('full source hash scan not complete')
PY
test ! -e "$S4_AT_EXPORT_ROOT" || { echo "Refusing existing export root; retain it and choose a new path" >&2; exit 2; }
mkdir -p "$S4_AT_EXPORT_ROOT"
tar -cf "$S4_AT_EXPORT_ROOT/clean_train.tar" -C "$S4_AT_CLEAN_ROOT" images/train labels/train
tar -cf "$S4_AT_EXPORT_ROOT/clean_val.tar" -C "$S4_AT_CLEAN_ROOT" images/val labels/val
tar -cf "$S4_AT_EXPORT_ROOT/s4_poison_train.tar" -C "$S4_AT_S4_PARENT_ROOT/dataset" images/train labels/train
tar -cf "$S4_AT_EXPORT_ROOT/s4_provenance.tar" -C "$S4_AT_S4_PARENT_ROOT" materialization_manifest.json dataset_integrity.json noise/checkpoints/step_009144_epoch_24_cursor_00000.pt
cp -- "$S4_AT_DATA_MANIFEST" "$S4_AT_EXPORT_ROOT/cloud_data_manifest.json"
(
  cd -- "$S4_AT_EXPORT_ROOT"
  sha256sum clean_train.tar clean_val.tar s4_poison_train.tar s4_provenance.tar cloud_data_manifest.json > SHA256SUMS
  for file in clean_train.tar clean_val.tar s4_poison_train.tar s4_provenance.tar cloud_data_manifest.json; do sha256sum "$file" > "$file.sha256"; done
)
echo "Uncompressed tar exports created. Source images/labels were not modified or re-encoded. No upload was performed."
