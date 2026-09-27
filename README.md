# Recovered original B2 source

This branch archives the first B2 implementation recovered from source commit
`ab594ab77496c32938d9e207e56b5fb20c8290d4`. It is a new publication branch;
existing object-B branches remain unchanged.

The code implements a frozen reference plus a differentiable one-step adaptive
learner. The original configuration uses inner learning rate 1e-4,
lambda_adapt=1, lambda_nt=0.25, frozen batch 16, and 40 noise epochs.
Production entry points are `scripts/train_noise_once.py`,
`scripts/materialize_noise_once.py`, and `scripts/run_isolated_victim.py`.
Historical shell scripts retain their original isolated server paths and RunIDs;
this archive does not authorize rerunning those scripts or overwriting outputs.

This first version predates the final 24-epoch budget and person-Recall reporting
fixes in `68e66f1afe63d61a55dab4f939fe7c0f895444b0`. Final B2 metrics belong to
that later implementation and are not results for this restored first version.
The archive has not been retrained. `provenance.json` records the exact SHA256
of every published source/config/test/script file, each byte-identical to the
recovered snapshot. Weights, datasets, output directories and the original
repository's asset-bearing Git history are excluded.

See [source provenance](experiments/b2_original_recovered_20260927/provenance.json).
