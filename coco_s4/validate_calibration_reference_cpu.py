"""New canonical-reference production function: synthetic native CPU probe."""
import copy
import json
import os
from .runtime import ROOT, load, now, sha, write_once


def main():
    output=ROOT/'execution'/('calibration_reference_cpu_'+now().replace(':','').replace('+','_'))
    output.mkdir(exist_ok=False)
    os.environ['YOLO_CONFIG_DIR']=str(output/'settings')
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from .phase_calibration import load_teacher, canonical_equivalence_probe
    from .calibration import collect
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer
    torch.set_num_threads(1);torch.manual_seed(0)
    old=load(ROOT/'execution/CALIBRATION_NATIVE_LOADER_CPU_VALIDATION_20261025.json')
    from pathlib import Path
    checkpoint=Path(old['retained_output'])/'random_native_ema_last.pt'
    model,before=load_teacher(checkpoint,sha(checkpoint),torch.device('cpu'))
    cfg=yaml.safe_load((ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml').read_text())
    images=output/'images';labels=output/'labels';images.mkdir();labels.mkdir()
    path=images/'000000000000.png'
    Image.fromarray(np.random.default_rng(0).integers(0,256,(640,640,3),dtype=np.uint8)).save(path)
    (labels/'000000000000.txt').write_text('0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n')
    seed=copy.deepcopy(model)
    trainer=LegacyKProtoTrainer(cfg,cfg['methods']['legacy_kproto_ret'],torch.device('cpu'),model)
    raw=collect(trainer,model,[path],labels,output/'collected')
    result=canonical_equivalence_probe(cfg,cfg['methods']['legacy_kproto_ret'],torch.device('cpu'),seed,
                                      path,labels,output/'baseline_statistics')
    assert result['all_sensitivity_sums_exact'] and result['reference_forward_calls']==2
    assert result['reference_actual_forward_tensor_devices']==['cpu']
    assert result['reference_actual_cuda_device_indices']==[]
    assert result['candidate_forward_calls']==1 and result['backend_flags_restored']
    assert result['formal_backend_flags_changed'] is False
    assert raw['target_count']==1 and raw['layers']['model.21']['nt_counts'][1]==1
    assert not torch.cuda.is_initialized()
    proof=dict(state='passed',UTC=now(),synthetic_images=1,imgsz=640,classes=80,
        reused_retained_random_EMA_checkpoint=True,actual_formal_checkpoint_used=False,
        frozen_model_hash=before,canonical_reference=result,GPU_initialized=False,
        source_sha256=sha(ROOT/'coco_s4/phase_calibration.py'),retained_output=str(output),
        official_metrics_generated=False,deleted_paths=[])
    write_once(ROOT/'execution/CALIBRATION_DETERMINISTIC_PROBE_CPU_VALIDATION_20261025.json',proof)
    print(json.dumps(proof))


if __name__=='__main__':main()
