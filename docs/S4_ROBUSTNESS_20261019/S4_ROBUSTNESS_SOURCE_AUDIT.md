# S4 原始来源核验

只读 RRCTL 实测已通过：唯一noiseRun s4_tfss_utility_seed0_20261017_v1；原始S4数据 /project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/s1_csb_20261014/s2_s3_counterfactual_20261015/s4_tfss_utility_20261017/outputs/s4_tfss_utility_seed0_20261017_v1/dataset。

{
  "parent_noise_checkpoint_sha256": "4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5",
  "parent_dataset_integrity_sha256": "3eece1fc62df963f7d4aa15d378395273bd3963f812c0f7764ea9115dc82fe82",
  "parent_materialization_manifest_sha256": "194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36",
  "last_weight_sha256": "cc1b03eca939732599817abb97150907463ab10913456afa6cc6ae473b4bbdc3",
  "s4_evaluation_sha256": "3b8da141dd366ea5dfde97fb7b56224abde61e9cd2931ad296a96d42482f1280",
  "victim_csv_rows": 200,
  "counts_verified": true,
  "runtime_versions": {
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "ultralytics": "8.4.150",
    "numpy": "2.4.6",
    "Pillow": "12.3.0",
    "opencv-python": "5.0.0.93"
  },
  "native_trainer_sha256": "2eda410e0bfb8b65cb46eb975c465c5b62b0f3a3dad5a54967d6e9e1fcabecc7"
}

Noise scientificsource b1e29510e7b1ed5fa0fe361a27132d8790e65c14；Victim修复运行ab626917f20917e58cf30033ac99df00938cb561。恢复pipeline报表KeyError不等于训练失败；最终200轮和评价文件身份已核验。此次基础代码取686eaac结果修复归档；不取S5实现。完整逐文件来源复核仍须在正式数据生成前运行，不能把目录计数冒充逐图SHA审计。
