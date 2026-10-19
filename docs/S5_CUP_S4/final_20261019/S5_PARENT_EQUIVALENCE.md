# Actual S4/off and neutral-CUP equivalence

{
  "source_commit": "e29d4662465159b5a1d1d7a5fd6c19c545f764e8",
  "smoke_path": "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/smoke_summary.json",
  "smoke_sha256": "fcead8be39d81fa45c4873097592fea80bd24790ac228627d743259fc32b8c1d",
  "max_difference": 0.0,
  "tolerance": 1e-07,
  "resume": [
    {
      "batch": 0,
      "passed": true,
      "additional_optimizer_updates": 5,
      "errors": {
        "trainer": 0.0,
        "optimizer": 0.0,
        "scheduler": 0,
        "rng": 0,
        "cup": 0.0,
        "cursor_state": 0
      },
      "cup_history_steps": 5,
      "journals_exact": true,
      "prefix_checkpoint": "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/batch0_CUP_split3/latest_continuation.pt",
      "resume_checkpoint": "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/batch0_CUP_resume5/checkpoints/step_000005_epoch_04_cursor_00016.pt"
    },
    {
      "batch": 1,
      "passed": true,
      "additional_optimizer_updates": 5,
      "errors": {
        "trainer": 0.0,
        "optimizer": 0.0,
        "scheduler": 0,
        "rng": 0,
        "cup": 0.0,
        "cursor_state": 0
      },
      "cup_history_steps": 5,
      "journals_exact": true,
      "prefix_checkpoint": "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/batch1_CUP_split3/latest_continuation.pt",
      "resume_checkpoint": "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/batch1_CUP_resume5/checkpoints/step_000005_epoch_04_cursor_00016.pt"
    }
  ],
  "native_AMP": {
    "state": "passed",
    "baseline_equivalence_max_difference": 0.0,
    "input_hashes_exact": true,
    "native_reference": "actual installed DetectionTrainer on identical S0 inputs; historical installed source unknown",
    "REAL_S4_VICTIM_BATCH_TEST": {
      "batches": 16,
      "actual_updates": 4,
      "skips": 5
    },
    "SYNTHETIC_OVERFLOW_TEST": {
      "batches": 14,
      "actual_updates": 2,
      "skips": 6,
      "synthetic": {
        "forced_batch": 11,
        "forced_after_actual_updates": 1,
        "multiplier": 1e+20,
        "recovered": true,
        "forced_optimizer_attempt_batch": 12
      }
    },
    "total_actual_batches": 40,
    "finite_loss": true,
    "finite_models_EMA_optimizer": true,
    "production_entrypoint_reached": true,
    "checkpoints_retained": [
      "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/AMP_trials/native_S0/retained_smoke_state.pt",
      "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/AMP_trials/repaired_S0/retained_smoke_state.pt",
      "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/AMP_trials/real_S4/retained_smoke_state.pt",
      "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s5_s4_cup_adaptation_20261018/outputs/s5_cup_p3_train32_smoke_seed0_20261019_v2/AMP_trials/synthetic_S4/retained_smoke_state.pt"
    ],
    "official_metrics_disabled": true,
    "artifact_ingest_disabled": true,
    "deleted_paths": [],
    "real_S4_legacy_trial_name_means": "actual S5 TRAIN32 paired PNG development dataset",
    "formal_S5_dataset": false,
    "noise_states": "two independent fixed16-image five-step trajectories",
    "validation_enabled": false,
    "smoke_only_val_save_plots_disabled": true,
    "formal_victim_hyperparameters_unchanged": true,
    "pre_AMP_resources": {
      "timestamp_unix": 1792350344.1083574,
      "sample_seconds": 5.0,
      "logical_cpus": 32,
      "cpu_affinity_cpus": 32,
      "cgroup_cpu_quota_cpus": null,
      "effective_cpu_capacity_cpus": 32.0,
      "loadavg_1_5_15": [
        6.578125,
        12.20751953125,
        20.01025390625
      ],
      "cpu_busy_percent_effective_sample": 7.21,
      "cpu_busy_percent_host_sample": 7.21,
      "cgroup_cpu_busy_percent_of_effective_capacity": 7.21,
      "cpu_headroom_logical_cores": 29.691,
      "io_wait_percent_sample": 0.04,
      "swap_activity_pages": {
        "in": 0,
        "out": 0
      },
      "mem": {
        "MemTotal_kib": 131118368,
        "MemFree_kib": 35530568,
        "MemAvailable_kib": 116797040,
        "SwapCached_kib": 32612,
        "SwapTotal_kib": 8388604,
        "SwapFree_kib": 7622996
      },
      "gpu": {
        "device_raw": "NVIDIA RTX A6000, 46068, 6313, 39152, 0",
        "compute_processes_raw": "1862863, /project/guoqinglong/miniconda3/envs/cfbdm/bin/python, 6006",
        "compute_pids": [
          1862863
        ]
      },
      "gpu_free_mib": 39152,
      "gpu_utilization_percent": 0.0,
      "disk_free_bytes": 1357205200896,
      "processes": {
        "d_state_count": 0,
        "foreign_uid_process_count": 596,
        "visible_uid_count": 24
      }
    }
  }
}
