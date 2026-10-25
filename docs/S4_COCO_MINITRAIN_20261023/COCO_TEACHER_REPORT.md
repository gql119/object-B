# Current accepted Teacher assessment — 2026-10-25

Teacher200 and its separate fixed native EMA last@200 full original val2017 5K
official COCO80 assessment are independently accepted. Person AP50=0.711971060380,
AP75=0.486437733439, AP50–95=0.462561678095. All80 macro AP50=0.395154267840,
AP75=0.283989529772, AP50–95=0.264909898204. Raw actual tensors were CUDA0/index0;
139 native data batches (maximum36, last32) plus one warmup. Checkpoint SHA
9749a7b6ceeb7b322316bdd97dfdde68f11641eef59fe6ba620a5e76a9521e3f unchanged.
No training/statistics/selection/tuning used validation. These are Teacher
measurements; they do not establish S4 effectiveness or full Experiment A completion.
Full per-class rows and immutable acceptance evidence are in
reports/TEACHER_EVAL_CALIBRATION_PRERUN_20261025. Historical progress below is retained.

# COCO Teacher formal training

Actual server verification: 2026-10-24T20:10:08Z (Hong Kong October25 04:10).

The single server-owned workflow `s4_coco_experiment_a_seed0_20261025_v3` automatically consumed the one scientific closure receipt and started the fresh Teacher200 phase. Actual runtime is `ee0b4256fdd344c5a275e6d15efc78cb7636d6e1`; subsequent preparation/report Git commits are separate from this immutable active release. Controller PID3797092/start349641321/uid1014 and formal child3800540/start349706115/uid1014 were independently verified alive. No normal stage was manually launched.

The first formal observation shows epoch1 in progress,57 native batches,51 actual SGD updates,6 legal AMP skips,consecutive skips0; model/EMA/optimizer finite. This is a dated progress observation, not completion of an epoch or an AP result. The fixed protocol is random-YAML YOLOv8n COCO80/Person0,200epochs,batch36,workers16,640,seed0,nativeSGD/AMP/original cosine schedule and base augmentation. Training uses all25000 clean training images and excludes validation. Primary checkpoint is fixed last@200. Official Teacher AP has not been measured.

The current-candidate GPU smoke completed20 native batches,14 actual SGD updates and6 legal AMP backoffs,exit0,zero validation-image/validator/EMA forwards. Candidate and separately instantiated native reference pre-update loss both equal487.11358642578125,difference0/tolerance0. Main-agent independent CPU audit checked86 source files and retained checkpoints; last optimizer/scaler and finite hook-free EMA were retained. No weights or images were pulled. Exactly one independent scientific review returned scientifically_correct/allow_run, with no Teacher blockers. Receipt9afbc4276ca92af2feb9bf4537f22c61c5c8e2a65c0723f1ea9b5c3beefcb02a was remotely read back exactly.

Failed v1 is retained: the strict binding rejected Windows backslash serialization before scientific work. Failed v2 is retained: its recorded processes disappeared after image caching with no Python exception or exit receipt; exact OS cause remains unconfirmed. The new production launcher uses ordinary independent systemd user scopes, verified by one actual CPU child; no priority, memory limit or OOM protection/policy override is used. No failed RunID was restarted.

Full data acceptance remains reused: fixed official25K train plus original cleanval2017 5K, all30000 image/label identities independently accepted. OriginalS4 8/4/0,StrictP5,SafeBG andTFSS remain frozen. Teacher completion, official post-training evaluation, real learned COCO statistics, Fourier24,export and matched Victims200 remain pending. The single controller has ordered gated phase slots; future component tests do not mark those phases ready. OP remains disabled and all old experiments remain accepted without reruns.
