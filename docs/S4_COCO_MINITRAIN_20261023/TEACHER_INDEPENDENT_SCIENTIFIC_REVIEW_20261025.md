## PRERUN-REVIEW-TEACHER-20261025

- Review mode: scientific_review
- Reviewer: Codex `/root/coco_teacher_scientific_review`, independent delegated execution. The review input was the lean packet, approved sources, committed diff, and referenced evidence; the main agent's conclusions were not supplied. Current model was inherited; an exact provider model identifier is not exposed to this reviewer.
- Result: scientifically_correct
- Decision: allow_run
- pre_run_result: pass
- Gated run: first formal COCO Teacher training only, using the exact command below.
- Code snapshot: `codex/s4-coco-minitrain25k-generalization` / `ee0b4256fdd344c5a275e6d15efc78cb7636d6e1`
- Diff base: `73724c5585633a6a7c95e6616f0a85f4b0c354f7`
- Approved basis: `docs/S4_COCO_MINITRAIN_20261023/USER_AUTOSTART_TASK_20261025.md`, `docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml`, and `execution/COCO_DATA_FINAL_INDEPENDENT_ACCEPTANCE_20261025.json`.
- Intent alignment: correct
- Canonical code location: correct
- Critical data flow: correct
- Computation sink effect: correct
- Runtime scientific state: correct
- Baseline/disabled path: correct, by the measured Baseline-Equivalence Probe described below.
- Experiment identity: correct
- Blockers: none in the reviewed Teacher phase.
- Validation gaps: none preventing this bounded scientific verdict. Teacher200 completion, downstream statistics/Fourier/export/victims, and final official evaluation are not claimed or passed by this review.

### Exact gated command

```text
/project/guoqinglong/miniconda3/envs/cfbdm/bin/python -B -m coco_s4.phase_teacher --release /project/guoqinglong/projects/s4_coco_experiment_a_20261025/bindings/s4_coco_experiment_a_seed0_20261025_v3/source_release.json --output /project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_v3/teacher200 --review /project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_v3/receipts/TEACHER_SCIENTIFIC_CLOSURE.json --smoke-evidence /project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_v3/teacher_smoke/smoke_summary.json
```

The release binds the candidate, config SHA256 `0b6ff803a0125748f6dc5aa2230da0cc4c14e7400cbdf6aeaf5f4189d9a110f1`, source release SHA256 `c520c67ea99effba2a81d79695d0102cd866c9475a8a7284793f6f0b3d426e1a`, accepted data root, native source hashes, and runtime versions. The output collision policy fails rather than resuming or incrementing. The exact scientific implementation files inspected match the committed candidate; untracked `coco_s4/coco_eval.py` was excluded.

### Scientific findings

1. **Data and label flow are aligned.** `phase_teacher.py:122-128` reads the bound config and official category mapping and writes a training-only YAML pointing at `/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023/images/train`. `teacher.py:114-146` checks all 80 names/order, Person0, the mapping hash, and the train route, then delegates training to the native YOLO loader. It does not substitute a Person-only sample or change GT. The production loader reports all 25,000 images. Data identity is supported by the approved prior acceptance: all 30,000 image/label identities checked, disjoint train/val IDs, and canonical annotation conversion. This review reused that acceptance rather than claiming a new image audit.

2. **Validation isolation covers native forced paths.** Null `val`, the empty validation sentinel, the forbidden validator, overridden `validate()` and `final_eval()`, and the EMA forward guard address the actual pinned native loop. In particular, native `engine/trainer.py:628-630` dispatches validation on the final epoch even when `val=False`; `teacher.py:191-203` replaces that dispatch with finite training-loss fitness only. Native final evaluation/optimizer stripping is replaced at `teacher.py:205-219`. The CPU final-epoch probe and current GPU audit both show zero validation images, validator forwards and EMA forwards. Reading accepted validation metadata hashes during release verification does not open validation images or feed validation examples into training. The negative training-loss fitness cannot change the fixed cosine schedule or select the primary checkpoint; `patience=0` disables early stopping in the pinned native implementation.

3. **Scientific values reach their intended native sinks.** The frozen config flows through `teacher_overrides()` into the instantiated `COCOTrainOnlyTeacher`: random YAML YOLOv8n, 80 classes, seed0, 200 epochs, batch36, workers16, imgsz640, SGD, AMP, cosine LR, close_mosaic100, lr0/lrf=.01/.01, momentum=.937, weight_decay=.0005, full dataset and base native augmentation. The pinned native source performs model creation with no input weights for `pretrained=False`, loader construction, preprocessing, optimizer grouping, native weight-decay scaling/accumulation/warmup, schedule, forward/backward, clipping, scaler update and EMA update. These computations are inherited rather than replaced. Native OOM batch reduction is rejected. The phase's local AMP preflight uses the random model and no downloaded weights; the real CUDA smoke separately demonstrates finite native loss and actual native SGD updates.

4. **Baseline reuse is numerically evidenced.** `phase_teacher.py:43-93` instantiates a separate native `ultralytics.nn.tasks.DetectionModel`, loads the candidate's exact pre-update weights, aligns CUDA memory layout, and executes both native loss calls on identical tensors/GT with RNG restoration. `teacher_smoke/baseline_equivalence.json` records sample_count36, imgsz640, nc80, seed0, no post-processing/NMS, reference and candidate loss sums both **487.11358642578125**, absolute difference **0**, tolerance **0**, and shared pre-update state SHA256 `da3da67a43ca838ca36846a01bc03a289b16037808099b0016d84c7bcaa03efb`. Thus the claimed native loss reuse passes its bounded probe. It does not establish a trained AP result or trajectory equality after 200 epochs.

5. **Checkpoint semantics preserve fixed last@200.** `teacher.py:177-189` removes only the passive EMA audit hook around saving and restores it afterward. Native `last.pt` retains its normal serialized EMA FP16 model and optimizer/scaler semantics; it is distinct from the extra FP32 unaveraged training-model snapshot. `ue_framework/s4_victim.py:51-68` writes model/optimizer/scheduler/scaler/Python/NumPy/Torch/CUDA RNG snapshots every 10 epochs including epoch200. Exact worker/prefetch continuation is explicitly unverified and automatic resume is disabled. `phase_teacher.py:157-174` requires native embedded epoch199, 200 CSV epochs and the epoch200 full-state snapshot for formal acceptance. Best.pt and training-loss fitness do not replace last.pt. The smoke's epoch0 checkpoint contains retained optimizer/scaler and finite hook-free EMA according to the referenced CPU readback audit; no claim that epoch200 already exists is made.

### Validation evidence

- `python -B -m coco_s4.validate_teacher_cpu`: recorded exit0; unchanged Teacher source SHA256 `e3954d420ffcf430aa672f58aebfb4f7c5043958f8bc1f88ddf3b5435327b024`; 72 synthetic CPU images, 2 native batches, 1 SGD update; native final-epoch/final-eval paths blocked; no validation/EMA forward; checkpoint state retained. Evidence: `execution/TEACHER_CPU_VALIDATION_20261025.json`.
- `python -B -m coco_s4.validate_teacher_contracts`: recorded exit0; 5 invalid data routes and 15 frozen argument mutations rejected; original 8 mathematical and 2 victim component sources unchanged. Evidence: `execution/TEACHER_CONTRACT_VALIDATION_20261025.json`.
- `python -B -m coco_s4.validate_phase_cpu`: recorded exit0; separate native CPU comparison 0.72416090965271 on both paths at tolerance0; candidate state restored; unsafe paths/collisions rejected. Evidence: `execution/PHASE_TEACHER_CPU_VALIDATION_20261025.json`.
- These earlier CPU results were inspected and their relevant source hashes verified, not rerun by this reviewer. Their synthetic data and differing local CPU package versions are not COCO experimental results.
- This reviewer independently recomputed the local release/evidence hashes: **86/86 source files, 7/7 pinned native source copies, 17/17 downloaded evidence files matched**, with zero mismatches. A committed-versus-working-copy diff of the scientific implementation/config files was empty.

### Current-candidate GPU smoke

- RunID: `s4_coco_experiment_a_seed0_20261025_v3__teacher_smoke`.
- Production command: the same interpreter/module/release as above, with output ending `/teacher_smoke` and `--smoke-batches 20`, without the formal review arguments.
- Candidate: `ee0b4256fdd344c5a275e6d15efc78cb7636d6e1`; exit0.
- Actual computation: **20 native training batches**, **14 executed SGD updates**, **6 native AMP overflow skips**; all recorded raw losses/model/EMA/optimizer states finite. Raw loss range: 485.83563232421875 to 500.2141418457031. Packet count24 includes the 20 training batches plus two reference/candidate pre-update loss forwards and two AMP preflight forwards; it does not mean 24 SGD updates.
- CUDA runtime: RTX A6000, Python3.11.16, torch2.7.1+cu128, torchvision0.22.1+cu128, Ultralytics8.4.150; seven relevant native source hashes pinned. Peak recorded CUDA allocation: 4,724,771,840 bytes.
- Official AP metrics and artifact ingestion disabled; all smoke checkpoints retained, no deletion. Native last SHA256: `04699f6630949453e9fcc5b82d5726d7bbc26979fb13bf0be3356d318c6892ce`.
- Evidence: `execution/teacher_gpu_evidence_20261025/teacher_smoke/smoke_summary.json`, `phase_spec.json`, `baseline_equivalence.json`, `train/TEACHER_TRAIN_ONLY_AUDIT.json`, `train/args.yaml`, `train/results.csv`, `amp_audit/AMP_SCALE_AND_SKIP_LOG.csv`, `jobs/teacher_smoke/exit.json`, and `INDEPENDENT_CPU_AUDIT.json`.

No experiment was launched, scientific code modified, or additional reviewer called by this reviewer. Scheduler, RRCTL, process scopes and artifact transport were excluded. This single verdict permits only the reviewed Teacher command/snapshot; it is not acceptance of future phases or a prediction that the resulting Teacher will meet its later detection-quality criterion.
