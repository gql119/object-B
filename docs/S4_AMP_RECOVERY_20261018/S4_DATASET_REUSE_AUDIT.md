# Existing S4 dataset reuse audit

Status: **PASS**. Fresh CPU-only RRCTL audit `87.64s`; no GPU forward and no noise generation.

- Source commit: `5cd8158e8312c1fcebb9fa26ac551acc3cd3a669`
- RunID: `s4_dataset_reuse_cpu_audit_seed0_20261018_v1`, RRCTL exit 0, 3-artifact contract finalized; bound workload PID 1656351, start ticks 292908556.
- Existing noise checkpoint SHA256: `4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5`. Dataset-integrity JSON SHA256: `3eece1fc62df963f7d4aa15d378395273bd3963f812c0f7764ea9115dc82fe82`. Materialization manifest SHA256: `194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36`.
- Dataset: 16551 train, 4952 clean validation, 6095 target images, 13256 Person GT changed.
- Labels and clean validation byte-identical: True / True. Non-target train images byte-identical: 10456. Duplicate/missing image IDs: 0.
- Pixel audit: outside max 0/255, global max 8/255, overlap max 4/255. Output-manifest image hashes and original 45 S0 source hashes passed.
- Original dataset fingerprint: `29b131dfca2e3636e9c47502961caa6297acf0450f646ff6409b7b3bfb21b3fa`. Runtime provenance: Python 3.11.16, PyTorch 2.7.1+cu128, Ultralytics 8.4.150, native trainer SHA256 `2eda410e0bfb8b65cb46eb975c465c5b62b0f3a3dad5a54967d6e9e1fcabecc7`. Historical contemporaneous trainer source hash remains unknown.

Inputs were read-only. No data or weights were transferred. Full result JSON and runtime provenance are retained in the local small-artifact archive beside this report. Next: fresh RRCTL resource gate, then the registered real CUDA AMP smoke.
