# S3 formal continuation, 2026-10-16

Change: S2 fully accepted/archived before S3. Only local report allowlist added to the previously unlaunched S3 RunSpec; source017591686efb3675ee8d55debf1e2cd7da4e0e17, reviewed command/config locks and all scientific settings unchanged. No additional reviewer or GPU admission rerun.

Result: rrctl ready passed; prelaunch no GPU process, 1427206361088bytes disk, MemAvailable122849556kB, no output/control/repo/stage collision. Unique S3 RunID s3_cnlp_cup_s0_eps8_person_seed0_20261015_v1 launched once09:42Asia/Hong_Kong. Actual09:47 observation noise_generation epoch1/19of9144steps,elapsed132.159368s,cursor304,19recorded key loss/gradient values finite,outside0,soleGPU9652796434MiB. S3 configuration lock bd89244e22795b85139d1dee793b13c1faf44306557c083f546639ec666b6fb8 matches actual identity and CUP config. Resume null; originalParent seed0, noS2noise initialization. No checkpoint yet before scheduled50steps.

Finding: Formal execution is observed, not merely ready/launch acknowledgement. Partial finite steps do not establish convergence, efficacy or causal mechanism. S2 primary target+1.1375043381pp/NT-.2575467315pp vsS0 did not authorize tuning; none performed.

Next: Monitor only this unique S3. Finish registered24/9144,fulltrain/cleanval/pixel/label/domain audits,randominit200,lastprimary/bestsecondary,VOC20/PUS/WCD/TCD3 and combined final reports. No new launches, deletions, weights/data transfers or environment/science changes. Keep independent initialization and original locks.

Evidence: execution/s3_formal_launch_20261016.json,execution/s3_prelaunch_resource_20261016.json,execution/s3_formal_ready_20261016.json,execution/s3_formal_latest_observation_20261016.json,docs/S2_TRAINING_REPORT.md,docs/s2_archive/LOCAL_ACCEPTANCE.json.
