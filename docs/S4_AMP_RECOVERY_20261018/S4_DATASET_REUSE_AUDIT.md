# Existing S4 dataset reuse audit

Status: **NOT_RUN this round — RRCTL connection timeout before any remote launch**. Previously reported passed receipts are provenance, not a substitute for the required fresh verification.

Prepared CPU-only script re-reads the actual finalnoise checkpoint and integrity/manifest hashes, checks every output image hash and label byte, all train/validation counts and IDs, original clean-data byte fingerprint, decoded target pixels under8/4/0 budgets, every13256PersonGTchanged and non-target image bytes. It uses original production decoding/rectangle-domain helpers and one CPU thread; it performs no GPU/model forward, image export or noise optimization.

Expected original noise SHA256 `4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5`.
Expected integrity SHA256 `3eece1fc62df963f7d4aa15d378395273bd3963f812c0f7764ea9115dc82fe82`.
Expected manifest SHA256 `194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36`.

Planned unique audit RunID `s4_dataset_reuse_cpu_audit_seed0_20261018_v1` is unlaunched. No audit receipt exists yet. Resume via RRCTL only after fresh resources and all lifecycle paths/bindings pass. A passed receipt is code/hash-bound; formal recovery also re-reads train image/label and validation byte hashes before training.
