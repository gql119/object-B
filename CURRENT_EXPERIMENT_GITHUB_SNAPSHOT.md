# Current experiment code snapshot — 2026-10-23

This is an isolated, code-only archive of actual server Git source. Each component is a separate package, not a merged scientific method. Server Git trees were clean with no untracked files; every included server source file was SHA-256 compared with the corresponding immutable Git blob. No running script was edited, no experiment was launched, and no COCO adaptation existed at snapshot time.

The server repository origins are RRCTL source bundles; their commit identities map to the canonical repository https://github.com/gql119/object-B. The source manifest records each actual commit and branch. Documentation publication HEADs are separate from runtime commits.

- original_s4_generation: original Safe-Soft05 + TFSS-Utility; VOC Person=14; 8/4/0 budget, overlap mask .5. This is the only permitted scientific parent of the later COCO experiment.
- original_s4_victim: AMP lifecycle repair and matched evaluation, separate from generation source.
- full_generation/full_runtime: Full ablation and lifecycle source. Its full-box support MUST NOT enter the COCO parent.
- old_robustness: eight accepted defenses and versioned reporting source.
- extended_runtime: currently bound selected-original-S4 Cutout/CutMix/MixUp successor; OP disabled and unapproved.
- s5: separate CUP prototype/smoke source. It is NOT a COCO parent and does not establish a completed formal result.

All source is under components/. Configuration files are redacted, non-runnable templates under templates/. PATH placeholders must be resolved from private audited runtime configuration before any run. Original configuration hashes and archive hashes are distinguished. Version information and native source hashes are in SNAPSHOT_MANIFEST.json. The native trainer's upstream Git revision is unknown; exact installed source files are hash pinned.

Excluded: VOC/COCO images, poison data, model weights/checkpoints, training logs, Python environments, cache, connection profiles, credentials and unrelated historical reports. These remain at their existing stores; original artifact identities are retained in configuration hashes and prior accepted result branches. There were no uncommitted server source changes to preserve. S5 is archived only to keep method lineages distinct.

Historical launch scripts are archival source, not safe default commands: never execute reset/cleanup paths. For future remote work use RRCTL and new unique outputs. COCO GPU work must wait until all currently authorized runnable robustness pairs are ACCEPTED, reports archived, actual training roots absent, and normal source/data/review/capacity gates pass. OP approval remains independent.

Snapshot completion additionally requires successful GitHub ref publication and remote commit/tree readback; see the external publication receipt. This document alone does not claim Push success.
