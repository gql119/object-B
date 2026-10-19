# P3/P5 TRAIN32 short comparison

Change: frozen TRAIN32 original32 images,65PersonGT,46NTGT; seed0; two independent16-image batches, originalS4 reference/P3 baseline/P5 baseline, each5updates/EOT2 and2 post-step EOT observations without a sixth update. Lambda_ret=.25; no CUP, no victim, no official AP.

Result: six trials completed,30updates, sourcebb5cd1e, exit0. S4/P3 equivalence maximum0; all72 total Fourier gradients finite, all6 Frozen/Adaptive hashes unchanged. Actual640-input shapes P3[16,64,80,80],P5[16,256,20,20]. Original8/4/0 and outsidezero assertions execute in each actual S4Trainer forward; no new PNG materialization was run in this layer-only diagnostic. S5 formal PNG audit remains NOT_RUN.

Finding: P5 final same-Frozen GT proxy improves in batch0, worsens in batch1; preservation-gradient magnitude and opposing protection projection are substantially larger. Class and object-size signals are mixed. Select P3 per StageC rule. Static initial favorable P5 cosine alone is insufficient.

Evidence: layer_probe_20261019/P3_P5_ANALYSIS.json; full trace, per-class gradient, original-ROI valid/included counts/size/channel-normalized drift, nativebox/BCE19/DFL delta and final per-class/batch CSVs; CHECKPOINT_METADATA.json. Raw eight artifacts and SHA/size manifest remain at local s5_small_20261018/<RunID>, remote checkpoints retained. All counts are descriptive and do not imply victim AP.

Next: P3 CUP-S4 adaptation, exact no-op and actual-path current-source validation, single independent S5 review; formal S5 not launched.
