# GitHub backup and change record

Date: 2026-10-30

## Pre-adaptation source and worktree separation

- Existing object-B published snapshot used as the backup parent: `64daca67003b3eab2321c7bc22775a90a95d9842`.
- Active Original S4 / AT runtime source remains `51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb`; this source was not changed.
- The active AT checkout had pre-existing uncommitted monitor/status files. The F: research checkout and a separate robustness-results checkout also had unrelated modifications. None was staged or copied.
- The backup was made in an isolated worktree from the existing object-B commit. The new backup commit adds exactly one provenance receipt; it does not rewrite parent history or change parent blobs.

## Backup publication

- Repository: [`gql119/object-B`](https://github.com/gql119/object-B)
- Branch: `codex/s4-ue-purification-backup-20261030`
- Parent: `64daca67003b3eab2321c7bc22775a90a95d9842`
- Backup commit SHA: `6bbf083ee30037abb872ea47f0df3ee96fbc922d`
- Normal non-force push completed successfully.
- Explicit `git ls-remote` readback returned `6bbf083ee30037abb872ea47f0df3ee96fbc922d` for the branch; this exactly matches local HEAD.
- New path in that commit: `docs/S4_UE_PURIFICATION_DEFENSES_20261030/GITHUB_BACKUP_RECEIPT.md`.
- Existing parent source blobs were reused without changes. The commit adds no dataset image, label, checkpoint, weight, profile, credential, token, or model cache. A tracked-tree scan found no private-key or GitHub-token pattern. Historical environment-specific path strings occur in already-published parent documentation; they were not introduced or changed in this backup commit.

## Adaptation branch and changes

- Independent branch: `codex/s4-ue-purification-defenses-20261030`.
- Base: the verified backup commit above.
- No Original S4 generator, Fourier/TFSS implementation, accepted VOC data, AT code, active RunSpec, or checkpoint changed.
- Added shared metadata-only pixel-purification GT/identity contract: `defenses/common/pair_contract.py`.
- Added unit tests for shared contract behavior: `defenses/tests/test_pair_contract.py`.
- Added explicit not-ready registry: `defenses/defense_readiness.json`.
- Added source-fidelity, readiness, backup/change, and experiment-plan reports under this folder.
- Added pinned source inventory for 43 upstream text files, with Git blob SHA and license note, in `UPSTREAM_SOURCE_INDEX.json`.
- No upstream implementation source or third-party weight was copied into this branch. OP/LE pinned trees have no root license file; AVATAR has component licenses; D-VAE includes MIT. These constraints are recorded in the source index/report.

## Scope and remaining limitation

The backup proves a recoverable, remotely verified object-B source snapshot; it does not mean the four purification defenses are implemented. The adaptation branch is preparation-only and all method-specific readiness flags remain false. CPU contract tests do not validate model loading, actual VOC image transformation, semantic/GT alignment, CUDA behavior, or formal training. The two latest RRCTL health calls failed to parse worker output, so the accompanying AT status is explicitly labeled as the latest local state-file observation rather than live server confirmation.
