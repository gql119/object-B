# CALIBRATION independent scientific review

- Review mode: scientific_review
- Reviewer: /root/coco_calibration_scientific_review; independent=true; reviewer_count=1.
- Result: scientifically_correct
- Decision: allow_run for CALIBRATION only.
- Code snapshot: 4679cef71843c13feb9a7604e1acc52f032e6d44
- Packet canonical SHA256: d362397b1019d555cff23e55cedef815a32808a84f7037c474bdf21e74856ebd
- Release SHA256: 5e16ca9f00af6e827cec3efea66b0ee930d90301ec95ac2d73a8b58ab04960ce
- Phase config SHA256: c8c732494757d394bf99ddd8cdcc5ff68cd94eb8a9d8c60bf011e396e76b2ccb
- Previous receipt SHA256: 69655e8c89aa282a5ace5eff89b45412d9631f76409d269d778728c1df868b51
- Teacher checkpoint SHA256: 9749a7b6ceeb7b322316bdd97dfdde68f11641eef59fe6ba620a5e76a9521e3f
- Blockers: none.
- Validation gaps blocking this calibration run: none.

## Gated command

/project/guoqinglong/miniconda3/envs/cfbdm/bin/python -B -m coco_s4.phase_calibration --release /project/guoqinglong/projects/s4_coco_experiment_a_20261025/releases/coco_calibration_20261025_0825z_detprobe/source_release.json --output /project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_v3/calibration --registration /project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261025_v3/receipts/calibration_READY.json

## Scientific assessment

### CAL-SCI-01 — Intent alignment: correct

The phase computes clean-training statistics for Original S4 COCO80 Person0 using the accepted fixed native EMA last@200. It does not optimize parameters, select checkpoints, use Teacher AP for method decisions or open validation images for method statistics.

Sources: coco_s4/phase_calibration.py:39, coco_s4/phase_calibration.py:77, coco_s4/phase_calibration.py:148, coco_s4/phase_contract.py:20.

Evidence: execution/coco_calibration_20261025_0825z_detprobe/source_release.json, execution/p/coco_calibration_20261025_0825z_detprobe/result/phase_spec.json.

### CAL-SCI-02 — Canonical code location: correct

Native EMA stays a DetectionModel, converts serialized FP16 to canonical FP32, enters eval and is frozen. P3 model.15 and P5 model.21 hooks feed unchanged gt_evidence, select_channels, class_subspace and select_strict_tsfl mathematics. Relevant ue_framework code is unchanged across the supplied review diff base and reviewed commit.

Sources: coco_s4/phase_calibration.py:148, coco_s4/calibration.py:33, ue_framework/methods/base.py:45, ue_framework/methods/semantic_coordinates.py:23, ue_framework/methods/tausb_universal.py:192.

Evidence: execution/CALIBRATION_NATIVE_LOADER_CPU_VALIDATION_20261025.json, execution/CALIBRATION_COMPONENT_CPU_VALIDATION_20261025.json.

### CAL-SCI-03 — Critical data flow: correct

The formal list is the hash-bound accepted 25K train manifest; each image and label SHA is checked during collection. Before acceptance the phase checks 25,000 images, 55,096 Person GT, 181,475 total GT, all79 NT classes and the complete digest. No-Person/empty-label images stay in that list. Sensitivity is per-GT abs-gradient spatial/batch mean, accumulated in canonical order then divided by object counts. Per-class bbox pooling uses original centered letterbox and area-mask geometry.

Sources: coco_s4/phase_calibration.py:102, coco_s4/phase_calibration.py:460, coco_s4/calibration.py:33, ue_framework/methods/semantic_coordinates.py:37, ue_framework/methods/legacy_kproto_ret.py:106.

Evidence: execution/CALIBRATION_COMPONENT_CPU_VALIDATION_20261025.json, execution/p/coco_calibration_20261025_0825z_detprobe/result/smoke_statistics/raw_statistics.pt.

### CAL-SCI-04 — Computation sink effect: correct

K=8 background prototypes use the original LegacyKProtoTrainer._build_bank in a separate clean no-grad P5 context pass with original masks, seed0 and max_images0. GT-gradient features are not substituted for that bank. Eligible support remains support.sum()>10, with reported later optimizer steps 24*ceil(N/16).

Sources: coco_s4/phase_calibration.py:473, ue_framework/methods/legacy_kproto_ret.py:234, ue_framework/methods/legacy_kproto_ret.py:292, coco_s4/calibration.py:89.

Evidence: execution/p/coco_calibration_20261025_0825z_detprobe/result/smoke_statistics/eligible_support_manifest.json, coco_s4/configs/calibration.json.

### CAL-SCI-05 — Computation sink effect: correct

StrictP5 keeps .4/.6/.15/min4/top3 and requires exactly24 channels; insufficient pools, degenerate PCA or incompatible space blocks without relaxing gates. P3 keeps .75/specificity1/variance.9/minrank4/maxfraction.5, and P3 orth reference copies are byte-exact. All79 full/selected NT bases are emitted. runtime_config binds these generated files; semantic_identity hashes them, then the formal phase instantiates the actual B2SemanticTrainer and checks24D/79 bases/P3 orth-retention/rank8/alpha.5/active3 before acceptance.

Sources: coco_s4/calibration.py:95, coco_s4/phase_calibration.py:127, coco_s4/phase_calibration.py:485, ue_framework/methods/semantic_coordinates.py:127, ue_framework/methods/b2_semantic.py:291.

Evidence: execution/CALIBRATION_COMPONENT_CPU_VALIDATION_20261025.json, coco_s4/configs/calibration.json.

### CAL-SCI-06 — Runtime scientific state: correct

The actual current-commit primary smoke reached raw cuda:0/index0 tensors before evidence reduction on one real accepted Person train image, with finite output/statistics and unchanged frozen model. It used one primary plus three isolated comparison forwards, total4, with no parameter update, official metrics or ingest.

Sources: coco_s4/phase_calibration.py:402, coco_s4/phase_calibration.py:427.

Evidence: execution/p/coco_calibration_20261025_0825z_detprobe/result/smoke_summary.json, execution/p/coco_calibration_20261025_0825z_detprobe/result/status.json, execution/CALIBRATION_CURRENT_GPU_SMOKE_LOCAL_AUDIT_20261025.json, execution/rrctl_state/coco_calibration_20261025_0825z_detprobe/last_status.json.

### CAL-SCI-07 — Baseline/disabled path: correct

The measured comparison is an extra isolated deterministic-cuDNN collection against independent original per-layer forward/GT-gradient loops. All960 present target/pooledNT/per-classNT sensitivity values match bitwise at tolerance0; both sides use actual CUDA0/index0 and identical fixed checkpoint bytes. Original backend flags are restored; primary smoke and formal collection keep them. This establishes exact equivalence in that explicit comparison setting, not default-backend bitwise repeatability or all79 real classes from one GPU image.

Sources: coco_s4/phase_calibration.py:247, coco_s4/phase_calibration.py:294, coco_s4/phase_calibration.py:427.

Evidence: execution/p/coco_calibration_20261025_0825z_detprobe/result/smoke_summary.json, execution/CALIBRATION_DETERMINISTIC_PROBE_CPU_VALIDATION_20261025.json, execution/CALIBRATION_COMPONENT_CPU_VALIDATION_20261025.json.

### CAL-SCI-08 — Experiment identity: correct

Packet canonical digest, reviewed commit, release SHA, frozen phase config, fixed Teacher checkpoint, prior teacher_eval receipt and exact official command agree. All5 retained smoke artifacts match declared SHA256/byte lengths; eight committed critical source/config files match release hashes. Collision policy is fail and the official gate requires the current smoke plus one review closure.

Sources: coco_s4/phase_calibration.py:173, coco_s4/runtime.py:65.

Evidence: execution/CALIBRATION_SCIENTIFIC_REVIEW_PACKET_20261025.json, execution/coco_calibration_20261025_0825z_detprobe/source_release.json, execution/p/coco_calibration_20261025_0825z_detprobe/artifact_manifest.json.

## Validation and evidence limits

All retained reports passed. Two synthetic all80 native CPU images match canonical gradient loops; synthetic79 artifact flow and incompatible24D blocking pass; random EMA/FP16 loading matches canonical state/raw/input gradients; current CPU isolated deterministic comparison passes.

Component/loader reports precede final probe-only change; collector mathematics and native loader are unchanged. Current CPU probe and current-commit GPU smoke cover final comparison path.

Actual current-commit CUDA smoke completed with exit0: one primary plus three comparison forwards, finite raw output/statistics, unchanged model and actual CUDA0/index0. All960 compared sensitivity values were bitwise equal at tolerance0 in the declared isolated deterministic-cuDNN setting.

- Formal all25K statistics, actual selected channels, real class PCA ranks, prototype count and generated B2 construction result are outputs of the calibration run and remain unmeasured before it. Their explicit fail-closed gates were reviewed.
- One-image CUDA evidence contains only its observed NT class. The all80-class CPU gradients and79-class artifact flow are synthetic component evidence, not COCO-wide empirical feasibility.
- The deterministic probe does not establish default-backend bitwise repeatability. Original enabled=true, benchmark=false, deterministic=false, allow_tf32=true flags are restored.
- No experiments were launched or scientific source files edited by this reviewer. Existing CPU/GPU evidence was inspected. Local reviewer interpreters lacked torch; no tensor deserialization or numerical rerun is claimed. The retained raw archive metadata and bytes were checked with standard library tools.
- The packet's local console.log path is absent from the pull directory. Completed exit-code evidence, production status, retained raw-statistics archive and matching artifact hashes provide scientific smoke evidence; console transport is outside this scientific verdict.

All currently evaluable scientific findings are recorded once. No final AP, efficacy, or later-phase readiness is inferred.

Machine-readable result: CALIBRATION_INDEPENDENT_SCIENTIFIC_REVIEW_20261025.json

