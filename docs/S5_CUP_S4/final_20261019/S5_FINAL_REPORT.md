# S5 final result

{
  "schema": "s5.local-acceptance.v1",
  "accepted": true,
  "run_id": "s5_tfss_utility_cup_p3_seed0_20261019_v1",
  "source_commit": "e29d4662465159b5a1d1d7a5fd6c19c545f764e8",
  "config_sha256": "afdcd04e925d73f50eeb9d450bdbb259195d1500f61f1e46ecd08dd66807f0c8",
  "noise_checkpoint_sha256": "63d68cc8c50afb35658a412cfdfafea0b89a5bc3222f53057bbaafb28fc780ee",
  "victim_completion": {
    "epochs": 200,
    "rows": 200,
    "last_sha256": "a33b864348bedfa4fa60feccdd420ea998faef410ce9025a89eee508ff2d291c",
    "best_sha256": "2e7c195e233476c50cea8df025820978e44d0c511857b0a60e3e8991fde41997"
  },
  "noise_steps": 9144,
  "noise_epochs": 24,
  "CUP_steps": 9144,
  "full_dataset_audit_passed": true,
  "all_class_metrics_complete": true,
  "criteria": {
    "Person_AP50_within_preregistered_limit": true,
    "NT_AP50_above_literal": false,
    "WCD_below_literal": false,
    "TCD3_below_literal": false,
    "NT_AP50_improved_vs_actual_S4": false,
    "WCD_improved_vs_actual_S4": false,
    "TCD3_improved_vs_actual_S4": false,
    "NT_AP75_improved_vs_actual_S4": false
  },
  "research_verdict": "MIXED_OR_REJECTED_BY_PREREGISTERED_CRITERIA",
  "unfavorable_NT_classes": [
    {
      "class_name": "bicycle",
      "AP50_delta": -0.016792262832639793
    },
    {
      "class_name": "bird",
      "AP50_delta": -0.014653433656207193
    },
    {
      "class_name": "boat",
      "AP50_delta": -0.019994677690452045
    },
    {
      "class_name": "car",
      "AP50_delta": -0.0003344808316485537
    },
    {
      "class_name": "cat",
      "AP50_delta": -0.01994207015279237
    },
    {
      "class_name": "dog",
      "AP50_delta": -0.01390139892492881
    },
    {
      "class_name": "horse",
      "AP50_delta": -0.004309828174949182
    },
    {
      "class_name": "motorbike",
      "AP50_delta": -0.031262052149556574
    },
    {
      "class_name": "pottedplant",
      "AP50_delta": -0.014423536122916136
    },
    {
      "class_name": "sheep",
      "AP50_delta": -0.024973083675729146
    },
    {
      "class_name": "tvmonitor",
      "AP50_delta": -0.0019382023202866305
    }
  ],
  "single_seed_not_significance": true,
  "automatic_extra_runs": false
}

P3 selected from matched TRAIN32 evidence; S5 changes only detached CUP-S4 class allocation of the original Frozen P3 L_ret. Original TFSS-U/.85, all other S0/S4 losses and weights, Fourier/TSFL/Frozen/Adaptive/SafeBG/budgets/data/native AMP/victim protocol are unchanged. No validation AP tuning or extra seed.

StageB and StageE train proxies do not establish full victim efficacy. Formal last@200 is primary and best is secondary. Differences are descriptive for one seed; no statistical significance, causal interpretation, or proof of layer universality. Negative classes and missing historical optional aggregate Recall/Precision remain explicit.

See complete aggregate/VOC20 CSVs, CUP weights/gradient audit, frozen config and raw SHA-bound artifacts. A positive result only makes S5 a candidate for the main method; this run does not establish robustness over seeds. Stop after this registered run; no automatic tuning or other experiment.

1. Layer: P3, because P5 train-proxy signals were mixed and its opposing protection gradient was larger.
2. Layer evidence: actual gradients/GT/size/5step audits justify the conservative choice, not a universal superiority claim.
3. CUP allocation: bounded normalized priorities and actual applied class weights are logged; classwise AP changes are in the VOC20 table and cannot alone prove the proposed mechanism.
4. Attribution: retention layer stays P3; the designed S4-to-S5 variable is CUP-S4. One seed does not exclude sensitivity or establish causality.
5. Person AP50=0.02258628164286698; S4=0.028158895413428475; change=-0.005572613770561494.
6. NT AP50=0.690105689344675; S4=0.6937870442308225; change=-0.0036813548861475143.
7. WCD change=0.017230562211597855; TCD3 change=0.003870650380017493.
8. Main-method assessment: MIXED_OR_REJECTED_BY_PREREGISTERED_CRITERIA; this remains single-seed descriptive evidence.
