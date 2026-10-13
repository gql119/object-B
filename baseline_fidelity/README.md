# Detection Baseline Fidelity — source snapshot

This directory contains the server-side Python source and tests for the EM-Det-Fidelity, SEM-Det-Fidelity, and EntF-Push-Det-Fidelity baseline implementations.

## Provenance

The files were copied from the isolated `source_full_v13_timewarning` source snapshot on 2026-10-13 (Asia/Hong_Kong). The runtime source hashes match the active SEM generation status manifest:

- SEM RunID: `sem_det_fidelity_seed0_20261013_new_full_bbox8_warningonly_v1`
- Protocol lock SHA-256: `14e3a817227e6f44badf5044ff8cb58bd9e6521919ae987e084d269ea0543edf`
- Schedule lock SHA-256: `32a3c0e71ed3abe59284178ec2889f089a9d25890c78f02800baae452965af62`

This is a code snapshot, not an experiment-results release. SEM was still running when captured; it makes no convergence or final-performance claim. The EntF implementation is included as a candidate, but its object-level ROI/centroid validity gate remains separate and has not been represented here as passed.

## Contents and limits

The Python implementation, helper scripts, and test source files are included under their original project-relative layout. `SOURCE_SHA256SUMS.txt` records the hashes of the copied source files.

Runtime locks/configurations, logs, generated perturbations, datasets, labels, checkpoints, model weights, temporary test artifacts, and Python bytecode/cache files are excluded. The code expects an experiment-specific lock and the user's configured data/model paths; inspect those inputs before any run. Some source files retain server-specific default path strings.
