# S1-CSB change log

## RECOVERY-001 — parent runtime bytes

Original Design: archive object-B commit0d43da4.

Observed Problem: archive checkout bytes differed from October4 runtime fingerprints (BOM/line endings, one default Kaggle path); active mathematical files matched once runtime bytes were restored.

Modification: recover exactly the 45 explicit Python/YAML runtime files in a new worktree; preserve bytes with .gitattributes. No historical file overwritten. Classification: BUG_FIX (source identity recovery, no mathematical change).

Scientific Justification: the evaluated parent's actual computation must be the reference.

Experimental Evidence: parent_remote_identity.json; numerical equivalence still pending.

Impact on Comparability: restores reference source identity; no CSB yet.

Final Decision: retain; do not claim loss/gradient equivalence before measured probe.

## PROBE-002 — absolute worker config binding

Original Design: parent equivalence subprocesses receive the same config argv.

Observed Problem: v1 failed before model computation because changing worker cwd to the new output made a relative config path unavailable.

Modification: resolve config/prototype paths before spawning workers; fresh probe RunID v2. Classification: BUG_FIX (launcher binding only).

Scientific Justification: both workers must consume identical existing config bytes.

Experimental Evidence: retained v1 rrctl console traceback; no loss or optimization result from v1.

Impact on Comparability: no loss, RNG or optimization setting changes.

Final Decision: retain; v1 failure preserved.

## CSB-003 — shared final poison binding

Original Design: override only final spatial projection, retaining inherited B2 losses.

Observed Problem: mechanism v1 completed native/fixed arms, then failed in CSB at B2's frozen-gradient diagnostic. MRO had cached the parent's old poison before S1 projected it; the frozen branch consumed the new tensor while Adaptive cached the old one. Traceback: autograd tensor unused.

Modification: S1's projection wrapper binds `_last_poison` to its final projected poison. Original B2/Adaptive/Frozen code is unchanged. New mechanism RunID v2. Classification: BUG_FIX, not a CSB parameter/design adjustment.

Scientific Justification: both parent branches must consume the same constrained support image.

Experimental Evidence: v1 console/failed status retained. Numerical closure requires the v2 production probe.

Impact on Comparability: corrects intended shared input; q10/q90, eta.25, gamma1, losses and optimizer unchanged.

Final Decision: retain pending passing production probe; formal blocked.

## RECOVERY-004 — committed blob byte identity

Original Design: restored worktree source bytes preserved by -text attributes.

Observed Problem: the production v3 config hash was133d3c rather than historical80cffe. Worktree bytes matched, but Git's unchanged-file index cache had retained LF blobs for5files despite new attributes (adaptive learner, orth reference, formal config, materializer, victim). Numeric tests passed because differences were line endings, but formal source/config identity checks would correctly refuse to run.

Modification: explicitly renormalize only those5files under -text attributes, preserving exact historical bytes in committed blobs; targeted test now checks both workingtree and Git HEAD bytes. Classification: BUG_FIX, transport/source identity only.

Scientific Justification: the deployed source and formal config must match historical identities.

Experimental Evidence: v3 actual config hash and local git-show SHA comparison;5matching committed blobs are required before the final probe.

Impact on Comparability: no source statements, configuration values, loss, RNG or scientific settings changed.

Final Decision: retain; v3 remains separate evidence and final exact-source admission uses fresh v4.

## PROJECTION-005 — actual old-map compatibility

Original Design: parent_fixed mode returned the parent tensor before consuming a supplied map; CSB used post-floor clamp.

Observed Problem: numeric no-op did not prove the task's stronger requirement that passing the actual old budget map through the new interface recovers historical amplitude scaling.

Modification: every mode now supplies a map. Exact old-map content (epsilon×original1/.5/0support) selects legacy amplitude scaling; other frozen maps use CSB clipping. Shared final poison is always bound. Add a separate csb-mode arm receiving old maps, compared to the actual original loop. Classification: BUG_FIX (required compatibility), not parameter tuning. No original source changed.

Scientific Justification: old overlap amplitude scaling cannot be reproduced by a generic clamp for sub-budget amplitudes; explicit content-based compatibility is needed, including legal noNT full8maps.

Experimental Evidence: passing final v5 probe required; prior disabled-mode checks remain separate.

Impact on Comparability: eta/gamma/Q/sensitivity unchanged; old-map numerical behavior and canonical losses are preserved.

Final Decision: retain pending measured old-map equivalence; formal still blocked.

## RESUME-006 — completed-checkpoint export

Original Design: resumable parent-equivalent Adam/EOT loop initializes the reporting status inside each optimizer step.

Observed Problem: the single independent reviewer on commit7a910ce identified B1: final-batch or epoch24-complete recovery performs zero new steps and reaches an uninitialized status variable. Prior3→5 replay did not cover epoch5 frequency freeze or production completion.

Modification: initialize status from saved cursor/diagnostics and retain the input checkpoint reference before entering the loop. Completion records actual restored epoch/steps. Add a production24-epoch diagnostic on two already-fixed train images: full export, epoch4→24 recovery, epoch23→last-batch interruption, zero-step last-batch recovery, zero-step complete recovery. Classification: BUG_FIX (checkpoint/reporting semantics); no loss, update, RNG, scheduler or spatial parameter changes.

Scientific Justification: exact recovery must also work when the registered optimization is already finished. The formal schedule remains24epochs/9144steps/6095images; two-image24-step fixture tests stage boundaries without claiming full-data generation.

Experimental Evidence: local5targeted tests passed; current-source GPU completion/frequency-freeze probe pending. The original independent scientifically_incorrect/do_not_run verdict is preserved in execution/independent_review_7a910ce.json.

Impact on Comparability: all parent numerical updates and CSB parameters unchanged. No second scientific review; main-agent closure requires measured zero differences and complete output.

Final Decision: retain fix pending production proof; formal remains blocked until B1 is closed.

## BINDING-007 — formal receipt/artifact roots

Original Design: formal command uses a local post-commit review JSON and passes the new worktree to the canonical semantic identity function.

Observed Problem: formalv1 failed before computation because rrctl source bundles contain committed files, not the untracked post-commit review JSON. Formalv2 then failed before cache/noise because the inherited identity function correctly rejects original TSFL artifacts outside the supplied new worktree root. Both failed RunIDs and logs are retained; neither updated perturbations or trained a victim.

Modification: explicitly transport the small review JSON to S1/receipts with SHA verification and bind an absolute CLI path; pass the authoritative read-only parent root to semantic_identity after separately checking all45new-source hashes. Include that exact parent identity in the bounded completion/recovery probe. Classification: BUG_FIX (deterministic identity/path binding only).

Scientific Justification: original semantic artifacts must be reused, not copied/rebuilt or silently replaced; new computing source is independently bound to the parent fingerprints. Receipts cannot be assumed present from permission-to-have-dirty-paths.

Experimental Evidence: v1 FileNotFoundError and v2 semantic-root ValueError; B1 completion probev1 passed with all parameter/checkpoint differences0 and diagnostic bytes identical. New-source parent identity and completion probe pending before fresh formalv3.

Impact on Comparability: no canonical mathematical file, artifact, parameter, loss, update or budget changes. No second reviewer.

Final Decision: retain binding corrections; preserve prior failures, use new RunIDs.

## CSB-008 — valid GT supervision coverage (independent candidate)

Original Design: every original GT competes in native TAL, and each NT GT must receive at least one positive before computing its object-balanced detection sensitivity.

Observed Problem: full frozen cachev3 failed at train2011_001260, GTindex5car, while200maps were committed. The valid GT measures168.96×60.16 pixels at640, overlapping a second car GT174.08×64. Native TAL resolves competing anchors to highest-IoU GT and can leave one valid object with no positive. No noise update or victim occurred.

Modification: new configs/s1_csb_gt_coverage_v2.json retains old locked config and all spatial settings. Only for an uncovered valid NT GT, call the same native TAL with that GT alone; retain its native soft scores, box/clsBCE19/DFL gains and object/class means. Existing covered GT assignments stay untouched. If the independent call still yields no positive/evidence, abort; no zero map, nearest-anchor hack or no-NT fallback. Classification: CSB_DESIGN_ADJUSTMENT, not a loss/update Bug Fix.

Scientific Justification: sensitivity estimates should include the supplied valid NT GT rather than erase it through assignment competition. This changes only frozen budget-estimation supervision; Safe-Soft05 losses/Teacher/Adaptive/optimizer and schedule remain unchanged. It may alter budget for previously unsupervised images and must be disclosed.

Experimental Evidence: retained full-cache failure and small annotation-size JSON. Require a new33-image probe (the same fixed32 plus failed train case), zero budget difference on the original covered cohort, finite nonzero recovered gradient, followed by full6095coverage admission. Not yet measured. No validation AP involved.

Impact on Comparability: old initial/locked configs, source commits and all old outputs retained. New candidate uses new config, cache and RunIDs; old partial cache is not relabeled under new identity. eta/gamma/Q/smoothing unchanged. Single original review is preserved; main-agent production evidence must document this later coverage correction, without inventing a second independent verdict.

Final Decision: candidate pending train-only coverage proof; formal noise/victim blocked.
