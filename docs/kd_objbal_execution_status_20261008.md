# Safe-Soft05-KD-ObjBal execution status

The single formal pipeline completed at 2026-10-08 14:10:45 UTC with exit
code 0. It ran in tmux `kd_objbal_formal_20261008`. Its isolated
server root is
`/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/kdcls_20261007/ue_project`.
The formal status is under
`runs_safe_soft05_kdcls_objbal_eps8_person_seed0_20261008/formal_safe_soft05_kdcls_objbal_seed0_20261008/pipeline_status.json`.
All formal stages completed. The final status, audit, checkpoint evaluation,
and matched comparison are preserved under
`experiments/kd_objbal_formal_20261008/`; see
`docs/kd_objbal_formal_result_20261008.md` for the result and limitations.

The fixed 16-train-query object-balanced calibration completed once under
RunID `calibration_objbal_seed0_20261008` and passed all preregistered gradient
gates. The bound lambda is `0.7160846249749698`; the calibration artifact SHA256
is `0ff854235663215da69b4fe04215c4a172885f165423b300f2864977903c9532`.
The locked config SHA256 is
`8107f4a3f7859b3d775cfa3c37d461a774bc379b2782206ec7b0c39b363fc741`.
The corrected deterministic parent-term equivalence probe reported zero
difference for all six checked terms, while retaining the original failed
probe artifact. The independent five-step smoke, pixel preview, and sanity
gate passed; gate SHA256 is
`bce7db06e43b0c28c348bf4cc2c132a4fc7f500e9f07ebea5d4b6252c996cf03`.
One independent implementation review returned `scientifically_correct` and
`allow_run`; review receipt SHA256 is
`9196b6e7f90416a9288b1d147ab5b7fab2b55456324b918d25fc8716c5b3475a`.

The 16-query calibration contained 24 valid non-target objects; every one had
ten TAL positives. On that set, object-balanced and positive-unit mean
reductions coincide algebraically. The observed gradient-tail change therefore
must not be attributed to the new reduction. The formal last.pt/best.pt
clean-validation evaluation is now complete and is reported separately.

The formal runner has an eight-hour poison-generation cap and guarded stage
sequence: 24 epochs/9144 steps, full 16551-image train materialization and
4952-image clean validation audit, random-initialized YOLOv8n 200-epoch victim,
then last.pt primary and best.pt secondary evaluation. The user requested
publication only after the experiment, so Git publication follows this
completed audit.
