# GitHub backup receipt — S4 source before purification-defense work

## Snapshot identity

- Repository: `gql119/object-B` (`https://github.com/gql119/object-B`)
- Backup branch: `codex/s4-ue-purification-backup-20261030`
- Parent snapshot: `64daca67003b3eab2321c7bc22775a90a95d9842` (already published object-B snapshot)
- Active Original S4 runtime source referenced by the live AT workflow: `51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb`
- This commit adds only this provenance receipt. It does not change Original S4, AT, experiment configuration, generated data, or checkpoints.

## Scope of this backup commit

The commit tree is the exact parent snapshot plus this one Markdown receipt. No files from another checkout were copied. The active AT worktree's pre-existing uncommitted monitoring files remain outside this commit. The local upstream-source reading copies remain outside this commit.

No dataset images, labels, model weights, checkpoints, runtime caches, profiles, credentials, tokens, or server connection details are added by this commit. Existing parent-tree blobs are unchanged; this commit does not re-add or rewrite them. A tracked-tree review found no private-key or GitHub-token pattern. Historical environment-specific paths occur in already-published parent documentation and were not introduced or changed here.

## Push verification

- Local commit: the Git commit whose tree contains this receipt; its SHA is recorded in Git metadata and the publication verification report.
- Remote branch head: checked after a normal non-force push by reading the exact branch SHA from GitHub.
- Verification: GitHub branch SHA must equal the local commit SHA.

