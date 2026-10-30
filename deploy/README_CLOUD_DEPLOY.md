# Original S4 VOC20 AT — AutoDL cloud preparation

This is a cloud-only deployment profile. It does not alter the school server's running experiment, the accepted Original S4 data, or the AT training algorithm. The four arm wrappers each launch exactly one arm on one host. Formal training remains blocked until source, all input/provenance hashes, CPU checks, real CUDA batch36/PGD10/resume evidence, one independent scientific review, and a path-bound accepted preflight are complete.

## Current AutoDL host snapshot

Read-only inspection after switching to GPU mode found RTX 4090 D, 24,564 MiB total, 24,081 MiB free, idle at the observation; root overlay 30 GB with 29 GB free; `/root/autodl-tmp` XFS data volume 50 GB with about 22.3 GiB free. The clean VOC directory already on that volume has 16,551 train and 4,952 validation images/labels; its counts and byte totals match the school Clean VOC manifest metadata, but a full per-file cryptographic comparison to the school copy is still pending. No S4 Poison training images or accepted S4 provenance bundle were found on this host. Do not start an arm until those are available and verified.

The source branch and report specify a 30/50 GB layout; use actual mount checks rather than trusting capacity labels. Set all code, Python environment, outputs, staging, and caches under the mounted `/root/autodl-tmp` volume. Do not add training data, venvs, or pip caches to the 30 GB overlay.

## Storage policy

The old 150 GiB free-space constant had no measured peak calculation. This cloud-only release replaces it with per-filesystem free-space checks:

- Before Python/environment installation: at least 15 GiB free on the configured data volume.
- Before staging Poison data: at least 12 GiB free on the actual data volume, covering archive plus expanded files and a 5 GiB reserve.
- Before an arm: at least 8 GiB free on the output volume (3 GiB one-arm output budget plus 5 GiB reserve), and at least 2 GiB reserve on each separate data/cache volume. Free space from separate mounts is never added together.
- During training: sample every 60 seconds and stop if the data-volume free space falls below 5 GiB. The last completed epoch's full state remains available; this guard does not change PGD, optimizer, or training updates.

The optimized one-arm output estimate is about 1.2–1.5 GiB from measured original checkpoint/log sizes: two rolling full states plus milestones every 25 epochs and final epoch 200; native `last.pt`, `best.pt`, and every-10-epoch weights remain. The 3 GiB run allowance is deliberately above that estimate. This is an estimate until a complete cloud arm is measured.

## Files currently on the data disk

The 50 GB volume currently holds about 28 GB. The largest entries are the top-level Git object database (about 22 GB; includes roughly 16.22 GiB loose objects, 4.57 GiB packed objects, and 513 MiB reported by Git as five temporary `tmp_pack_*` garbage files), the root `VOC_0712_Kaggle_Ready.zip` (2.23 GB), and the expanded clean dataset in `ue_project/VOC_0712_Kaggle_Ready` (about 2.3 GB). `ue_project/runs` is about 1.2 GB, mainly historical TAUSB checkpoints, logs, and detection runs. A nested `ue_project/autodl-tmp/.git` is about 910 MB. The current AT deployment does not use the TAUS mask folder (about 69 MB), but it is retained with the dataset.

No data was deleted. Future cleanup candidates, pending owner review and identity checks, are the ZIP if it is proven to be an exact duplicate of the expanded Clean VOC folder; the five Git `tmp_pack_*` entries only after confirming no Git operation owns them; old `runs` artifacts only after their experiment provenance is reviewed; and the nested Git checkout only if confirmed redundant. The large top-level Git object database is not a safe general cleanup target. Never delete the clean VOC copy or the only verified S4/provenance copy.

## Source and host configuration

Use the published cloud source commit recorded in the release report. Do not use the current school runtime commit or a moving default branch. Keep `S4_AT_EXPECTED_SOURCE_COMMIT` equal to the reviewed cloud release SHA. The cloud source closure is rechecked before paths are mapped and before formal launch.

1. Load `cloud_configs/S4_CLOUD_COMMON.env.example` and set `S4_AT_EXPECTED_SOURCE_COMMIT` to the published cloud Release SHA. Load exactly one arm config.
2. Put the source clone at `$S4_AT_SOURCE_ROOT`; put clean data at the existing `$S4_AT_CLEAN_ROOT`; stage accepted S4 Poison data and its provenance below `$S4_AT_S4_PARENT_ROOT`.
3. Run `deploy/cloud_storage_check.py --source-root "$S4_AT_SOURCE_ROOT" --phase data_stage` before transferring/extracting the Poison archive. Verify the full source SHA manifest and accepted provenance. Only after that may a transfer archive created for this deployment be removed, one named file at a time.
4. Run `deploy/configure_host.py` with the clean and S4 roots. It creates only the frozen legacy path aliases, refuses to replace existing paths, verifies accepted provenance hashes, and leaves `PROTOCOL.json` untouched.
5. Run `deploy/setup_env.sh`. It creates Python 3.11 and the pinned CUDA 12.8 packages in `$S4_AT_ENV_ROOT`; TMP, Conda, pip, Torch, and XDG caches are pointed to the data volume. It refuses an existing environment and retains an observed `pip freeze` receipt.
6. Run `deploy/preflight.sh` only after clean/S4/provenance files and their full SHA manifest are present. It records evidence only; it never writes an accepted receipt or starts training. Run the single independent review and main closure on that actual GPU evidence before creating a path-bound preflight receipt.

The accepted auditor requires Clean train, S4 train, Clean validation, and provenance on every host. The Clean VOC metadata currently matches the school manifest, but exact per-file SHA verification is still required. The S4 data bundle is missing on this host, and exporting/scanning the school server's active data has not been done in this preparation.

## Environment and GPU readiness

The current package contains observed direct pins, not a complete wheel-hash lock. `setup_env.sh` records the actual installed environment, but this alone is not scientific approval. Driver-reported CUDA 13.0 and the RTX 4090 D are only platform observations. PyTorch has not yet been installed on this host, so CUDA compatibility, actual Batch36 peak memory, full PGD10, native YOLO loss/GT, SGD, EMA, scaler, full-state save/load, resume, and checkpoint policy remain untested here. The old 24,000 MiB free gate is retained until the actual full probe measures memory; do not lower it from the GPU's name or label.

## One-arm commands

Each cloud machine receives one arm config and a unique RunID/output root. Formal command wrappers remain gated by the reviewed `ACCEPTED_PREFLIGHT.json` and use the native frozen protocol: seed0, random YOLOv8n YAML, batch36, imgsz640, workers16, 200 epochs, PGD10, inner FP32, outer AMP/native GradScaler, complete GT and native detection loss. No algorithmic setting is varied by the storage profile.

```bash
source cloud_configs/S4_CLOUD_COMMON.env.example
source cloud_configs/S4_CLOUD_CLEAN_AT4.env.example   # select exactly one per host
export S4_AT_EXPECTED_SOURCE_COMMIT='<reviewed cloud release SHA>'
bash deploy/setup_env.sh
bash deploy/preflight.sh
# After actual-source CUDA evidence, one scientific review, main closure and accepted path-bound receipt:
bash deploy/run_at4_clean.sh   # use the one matching host arm only
bash deploy/check_status.sh
bash deploy/evaluate.sh
```

The formal command block is illustrative and must not be run until every gate passes. `check_status.sh` is read-only. `evaluate.sh` verifies the already-produced `last.pt@200` metrics and does not rerun validation. Each host runs one arm; the four runs use separate hosts and independent output roots.

## Current blockers

- The host is GPU-enabled, but only one of four planned hosts is available.
- No S4 Poison dataset/provenance bundle or complete data SHA manifest is present on the cloud volume.
- The Clean VOC folder matches school metadata but has not yet been compared file-by-file by SHA256.
- The new storage/checkpoint source still requires local validation, a bounded actual CUDA probe, one scientific review, and a reviewed immutable release before formal readiness.
- No formal training has been started on this cloud host.