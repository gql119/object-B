# Selected original S4: final extended augmentation results

Cutout, CutMix and MixUp each completed a fresh matched Clean200 / original-S4200 pair. All three pairs and their small report archives passed independent acceptance. OP is **BLOCKED_PROTOCOL_APPROVAL**, disabled, and has no projection-model or Victim result. No OP effectiveness claim is made.

Selected Method: **S4_ORIGINAL**. Immutable runtime source `40d2e55914c586bba45d1ea1bf44b89689d20865`, config `e35f7119a929b678c2f074c64ebb4641cf25fa41d1963eae7a9f4bbecf262677`, Decision `630c8ad65d110c115a72dc7ea50d88c55a2a1ffaac79172d239b0d02a0e612c1`. Decision origin remains selected-v2/source8644/config97a4; its exact bytes were inherited, without reselection. Later documentation commits are not runtime source.

All values below are percentages; raw CSV/JSON retain signed fractions without clipping. Main evaluation is last.pt@200 on unchanged clean VOC20 validation4952; best.pt is secondary. Each defense uses its own newly trained Clean denominator.

|Defense|Person AP50 Clean / S4|NT AP50 Clean / S4|P|U|S|WCD|TCD3|
|---|---:|---:|---:|---:|---:|---:|---:|
|cutout|87.016 / 3.341|77.384 / 69.873|96.161|90.294|93.135|26.621|23.332|
|cutmix|86.477 / 4.179|76.630 / 68.934|95.168|89.956|92.489|28.749|24.849|
|mixup|87.276 / 9.464|78.607 / 69.096|89.156|87.901|88.524|30.758|24.921|

In this seed0 protocol, S4 retains strong Person suppression under all three augmentations. MixUp has the highest S4 Person AP50 and the lowest paired S among these three; it also has lower non-target retention. These are descriptive results, not statistical significance or a causal claim. Cutout/CutMix/MixUp are training-time data augmentations; OP would be a separately approved data-recovery defense.

Protocol: random-YAML YOLOv8n20, seed0,200epochs,batch36,workers16,AMP,nativeSGD,640,locked LR/scheduler/warmup/base augmentation. Cutout:1hole,p1,nominalarea.25,fill0; reject the entire candidate if any GT has >50% bbox occlusion,8 attempts then skip; retain full GT; all200epochs. Native Detect CutMix:p1,Beta1,3areas,primary zero GT intersection,donor>=.5visible/min2px. Native Detect MixUp:p1,Beta32/32,integer-instance concatenation. Native close_mosaic100 disables CutMix/MixUp in the last100epochs. No combined new defenses or classification CE label interpolation.

Verification: RRCTL completionhealth -> wait ->18small report pull, exit0; every archive byte/SHA checked. Independent single-thread CPU reread of131source/12native hashes,12actuallast/best checkpoint hashes/tensors,6complete200-row CSVs and embeddedlast histories,6native200 optimizer/scheduler/scaler/Python-NumPy-Torch-CUDA RNG states; no GPU initialized. All16551manifest rows and4952clean validation image/label links per arm verified;3current source-image/label samples per arm rehashed. Full original-S4 file scans are the retained server validation/acceptance receipts; this completion audit does not claim to have repeated every image-byte scan. Checkpoint files, image data and profiles were not downloaded or published. Legal AMP skips are retained and reported, not reclassified as failures or hidden by hyperparameter changes. No exact DataLoader/prefetch RNG resume is claimed.

Files: `RECOMPUTED_RESULTS.csv` adds NT AP75/AP50-95,all AP50,Person precision/recall; `server_archive/reports/EXTENDED_PER_CLASS_RESULTS.csv` contains all20classes; `server_archive/reports/METRICS_ALL_CHECKPOINTS.json` contains all12primary/secondary results; `CHECKPOINTS_AND_UPDATES.csv` preserves actual identities; `CLEAN_ABSOLUTE_CHANGE.json` reports absolute Clean change against the locked bare-Clean reference (only the matched Clean+D is the P/U/S denominator). `evidence/` contains actual arm/pair acceptance,run specs,CSV and source/data receipts.

OP protocol action: explicit human approval of a detection adaptation remains required. The proposed pooled instance64/shared class-agnostic basis/overlap averaging is a proposal, not an approved or standard-paper detection protocol. The first three results are archived independently and will not be retrained while OP waits.

COCO successor: the old three-pair prerequisite is now accepted and no own training root was found. COCO remains blocked by physical25Ktrain/5Kval image acquisition, new full orchestration/source binding, isolated evaluator dependencies and new-phase GPU/review/capacity gates. No COCO GPU job has started. Keep the same sole monitor; no automatic AT,extra seed,method change or second monitor.
