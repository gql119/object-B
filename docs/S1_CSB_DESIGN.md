# S1-CSB frozen spatial budgeting

Status: Stage 0 parent recovery and numerical equivalence in progress. No formal S1 run.

Authorization: user-supplied 2026-10-14 task at attachment `ccaf8790-eef0-4ba0-8504-6417c9cd758f/已粘贴的文本.txt`; full S1 is authorized only after all stated gates. No second full budget-matched control, S2, or changes to Safe-Soft05 losses.

Parent archive Git SHA: `0d43da4c76d316c7dbc588c6a1f7d252bb3f95c0` in object-B. The October 4 runtime deferred Git commit until after the experiment; no contemporaneous runtime Git SHA was recorded. Scientific identity is the runtime source/config/artifact SHA receipt, not an invented runtime Git SHA. The new worktree restores the exact runtime bytes for 45 Python/YAML files. Archive differences were UTF-8 BOM/line endings and one unused default Kaggle path; the formal YAML specifies the actual VOC root. See `s1_evidence/parent_remote_identity.json`.

Parent RunID: `b2_to_safe_soft05_eps8_person_seed0_20261004`, noise `noise_full_...`, victim `victim_...`. Last primary: Person AP50=.028993419332965346, Recall=.0006625441696113075, NT AP50=.6916980897616373, all AP50=.6585628562402036. Victim last SHA `43c24b8704b22af723a99c028e0dcb9f850d74fcb645f0d3baf26d8e68665c7f`.

Frozen Reference and each virtual Adaptive initialization share `voc20_surrogate.pt`, SHA `8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`. Frozen/Adaptive base state is unchanged; Adaptive clones parameters for a single differentiable SGD step, inner lr=.0001, batch1. Its model.15 clean-query target evidence and NT feature retention remain unchanged. Parent Adam updates 64 candidate Fourier coefficients, selects 16 bases after five warmup epochs, cosine scheduler, lr=.001, 24 epochs, batch16, 2 EOT: 9144 optimizer steps. Suppression parameter exists but is not optimized by this parent. No independent-pixel delta optimization.

TSFL P5/model.21 and ordered 24 channels are frozen: [0,6,10,29,41,68,77,86,89,112,116,125,128,133,140,167,173,184,220,229,237,238,239,240]. Safe-BG rank8/alpha.5/active3, cached projectors; orth coordinates model.15, worst3, overlap weight2, lambda_orth=.0005. BG/rel weights1, ret=.25, TV=.0001, budget penalty0, Adaptive weight1/NT .25. Existing full background prototypes SHA `12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7` are reused unchanged on server.

Parent composition applies tanh/JND, training scale/dropout, global clamp, straight-through minimum magnitude2.01/255, then multiplies by the fixed 1/.5/0 support. Therefore old overlap behavior is amplitude scaling, not merely a final clamp. Old-budget mode must execute the unchanged parent implementation and pass numerical equivalence. CSB mode will reuse the same latent variables, RNG, losses and target support, altering only final spatial feasibility projection. Original raw regularization input and masks remain parent-defined; CSB must not rescale raw TV or change support/query selection.

Proposed initial CSB: fixed clean-teacher NT-GT input gradient, RGB L2; Person-domain Q10/Q90 normalization, masked 3x3 mean smoothing, clip[0,1]; epsilon=(8/255) M_person [.25+.75(1-Ghat)]. Classwise/aggregated distinction and object/component reductions must be documented before implementation. Only no-NT-GT may use full bbox8 fallback. Constant-gradient handling must be deterministic and logged; invalid/missing/nonfinite NT supervision is an error, never zero-gradient fallback.

Sensitive maps and budgets are precomputed on train only, never dynamic poison input or clean validation; identity binds image/annotation/teacher/config hashes. PNG integer projection must obey the per-pixel budget, labels byte-identical, no-Person images unchanged, clean val4952 byte-identical. Fixed train diagnostics include uniform mean-budget control, which is NOT a second formal victim.

Formal victim remains random-init YOLOv8n200,640,batch36,workers16,SGD lr.01/lrf.01,momentum.937,wd.0005,cos_lr,close_mosaic100,AMP/cache,seed0. Primary last200, secondary best; unchanged parent validator defaults/NMS. No formal generation before parent/CSB/resume/budget/resource gates and the single scientific review pass.
