# Direct human authorization: COCO80 selected-dimension compatibility

Received 2026-10-25 in this chat. This authorization supersedes the prior 24D-only COCO block **only for selected-feature dimensionality**. The old failed workflow and calibration snapshot remain immutable.

Required result label: `Original S4 — COCO80 Dimension-Compatible Implementation`.

Preserve actual StrictP5 selection: `min(ceil(.15*C), N_eligible)`; actual C=256 and D=39. Retain every selected index, score, ranking and calibration identity. No truncation, candidate-fraction/safety-threshold/Top-3 sensitivity changes or layer changes.

Support actual selected D in Safe-BG and all consumers/artifacts/cache contracts. D comes from selected indices, never a literal39. Preserve VOC24 behavior and numerics; rank8, removal_alpha=.5, active_k3, ordered low-rank SVD projection and norm restoration, L_BG/L_rel/L_orth/L_ret, P5 Safe-BG and separate P3 Orth. All selected-space vectors/prototypes/NT bases/projections share the same coordinates. Keep effective-rank, orthogonality, finiteness and complete class checks; no skipped classes or silent rank fallback.

Frozen Original S4 spatial protocol: Person-exclusive8/255, Person/NT overlap4/255, outside0, TFSS eta=.85 and unchanged Q_t/R_nt, Fourier/Frozen/Adaptive/loss weights. Fixed official COCO minitrain25K, COCO80/Person0, original val2017 5K. This is not Full/S5/CUP or method tuning.

Required actual validation: old/new VOC24 Safe-BG numerical equality; COCO39 shapes/rank8; L_BG and L_rel gradients to Fourier; all79 NT categories in subspaces/ranking; bounded on-demand memory; related Original S4 regressions; independent P3 coordinates unaffected. Stop and report any material mathematical change, numerical mismatch or unverified VOC compatibility.

First hash/check retained full25K raw statistics, original39 selection, eligible support manifest, K8 prototype bank and accepted Teacher identity. Reconstruct only affected coordinates/bases/cache from retained statistics when sufficient. Do not repeat full calibration or retrain accepted Teacher. Explain any genuinely missing dependency before selective recomputation.

Use an independent COCO branch with a new protocol version. Preserve previous commits/failure records. Deliver COCO_SAFEBG_DYNAMIC_DIMENSION_AUDIT.md and numerical consistency report, commit/push to GitHub and verify exact remote commit. No force/remote-overwrite/credential disclosure.

Use the same sole monitor s4-robustness-existing-run-status and existing server control mechanism. Old serial controller stopped; do not restart/reuse the failed workflow. Create a uniquely identified, traceable recovery with accepted stages reused and no parallel GPU scheduler. Formal continuation requires actual dimensional compatibility/VOC regression/coordinate-cache identity/Fourier gradients/verified publication/unchanged8-4-0/no GPU conflict, plus applicable current-candidate CPU→bounded GPU→one scientific review→main closure gates. No redundant human confirmation.

Then automatically continue Fourier24 → poisoned25K export → fresh matched Clean200 → S4200 → original cleanval2017 evaluation → final report/publication. Keep original experiment configuration. Report actual COCO39 and VOC24 under identical StrictTSFL selection rules. Five-hour quota recovery remains authorized via the same monitor without reset credits.
