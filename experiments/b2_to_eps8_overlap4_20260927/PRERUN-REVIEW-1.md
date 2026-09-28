# PRERUN-REVIEW-1: B2-TO E8 budget ablation

- Review mode: scientific_review
- Reviewer: independent delegated reviewer, fork_turns=none
- Result: scientifically_correct
- Decision: allow_run
- Gated run: `bash scripts/run_b2_to_eps8_full_20260927.sh` in isolated tmux `b2_to_eps8_pipeline_20260927`
- Code snapshot: `codex/b2-to-e8-ablation-20260927` / `0cce7d6e787220fc4a4fd3b30c68bf4c61fc68c5`
- Diff base: `7906bc5fe4338eb40774412bb838b6ecefddd06a`
- Approved basis: user's strict half-budget B2-TO ablation request
- Intent alignment: correct
- Canonical code location: correct
- Critical data flow: correct
- Computation sink effect: correct
- Runtime scientific state: correct
- Baseline/disabled path: not_applicable (no equivalence claim)
- Experiment identity: correct
- Local validation: focused server unit tests (2 passed); py_compile; bash -n; config-diff validator confirms only epsilon and isolated output root differ.
- GPU smoke: `noise_sanity_b2_to_eps8_overlap4_5steps_20260927`, five steps, exit 0, finite L_total; eight decoded production PNGs satisfy target max 4/255, overlap max 2/255, outside-target 0. Official metrics and artifact ingest disabled.
- Blockers: none
- Validation gaps: none for admitting the E8 run; production full-data paired audit remains a runtime gate and must pass before victim creation.

Reviewer findings: E8 config keeps B2-TO method/data/victim settings fixed and changes epsilon from 16/255 to 8/255; the unchanged overlap scale 0.5 yields a 4/255 decoded overlap cap. Production materialization writes PNGs and `_full_data` validates decoded values, label bytes, stems, target coverage, and clean validation identity. The deterministic paired 256-person-image audit checks B2-TO labels and decoded perturbation statistics before the victim is imported or constructed. The pipeline runs noise, verifies it, materializes, runs preflight/audit, then trains the fresh random-init YOLOv8n victim; failures block downstream training.
