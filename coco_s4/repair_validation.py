"""Retained validation-only native parameter -> export link; never a formal dataset."""
from pathlib import Path

def export_link(trainer,cfg,method_cfg,params_path,image_path,output):
    import cv2,numpy as np,torch
    from ue_framework.data_utils import load_image_rgb_float,read_yolo_annotations
    from ue_framework.methods.legacy_kproto_ret import LegacyKProtoGenerator
    from ue_framework.methods.s4_tfss import model_hash
    from ue_framework.methods.tfss_spatial import spatial_weights
    from .phase_export import compose_tfss_utility_png,validate_decoded_pair,_save_png,_decode_png
    from .fourier_artifact import validate_native_params
    from .runtime import DATA_ROOT,sha,write_once
    output=Path(output)/'validation_export';output.mkdir(exist_ok=False)
    params=torch.load(params_path,map_location='cpu',weights_only=True)
    identity=validate_native_params(params,num_classes=trainer.num_classes,target_class_id=trainer.target_class_id,feature_layer=trainer.feature_layer)
    label=DATA_ROOT/'labels/train'/(Path(image_path).stem+'.txt')
    annotations=read_yolo_annotations(str(label));clean=load_image_rgb_float(str(image_path))
    before=model_hash(trainer.surrogate)
    generator=LegacyKProtoGenerator(cfg,method_cfg,trainer.device,trainer.surrogate,str(params_path))
    result=generator.generate(clean,annotations,seed=0,steps=24,eps=8/255,support_type='mask',image_path=str(image_path))
    support=result.support_mask
    from .canonical_s4_training import cpu_rng_isolation
    with cpu_rng_isolation(trainer.device):
        maps,records=trainer.s4_cache.get(trainer,[(clean,annotations,support)],[str(image_path)])
    q,risk,_=maps[0]
    candidate,png,weight=compose_tfss_utility_png(clean,result.perturbation,q,risk,support)
    # Same original final-generation context: a single original image, epoch23
    # final JND/tanh schedule. No claim of arbitrary padded-batch equality.
    old_weight=trainer.tfss_weight
    old_training_mode=trainer.is_universal_training
    with torch.no_grad():
        x=trainer._to_tensor(clean);s=torch.as_tensor(support,device=trainer.device)[None,None]
        trainer.tfss_weight=spatial_weights(q,risk,s[0,0],mode='TFSS-U',eta=.85)[None,None]
        # Compare two original FINAL-GENERATION constructions, not an original
        # stochastic training draw (which retains random scale/dropout).
        trainer.is_universal_training=False
        try:
            _,delta,poison,_,_=trainer._compose_delta_batched(x,s,torch.zeros_like(s),params['coords'],
                params['fourier_coeff'].to(trainer.device),params['suppress_small'].to(trainer.device),current_epoch=23)
        finally:
            trainer.tfss_weight=old_weight
            trainer.is_universal_training=old_training_mode
    native=poison[0].permute(1,2,0).cpu().numpy()
    max_abs_difference=float(np.max(np.abs(native.astype(np.float64)-candidate.astype(np.float64))))
    if not np.array_equal(native,candidate):raise ValueError('Native final-image S4/export construction differs')
    path=output/(Path(image_path).stem+'.png');_save_png(path,png,cv2)
    decoded=_decode_png(path,cv2);clean_u8=np.rint(clean*255).astype(np.uint8)
    from .data import audit_pixels
    audit=validate_decoded_pair(clean_u8,decoded,support,audit_pixels)
    target=output/label.name
    with target.open('xb') as stream:stream.write(label.read_bytes())
    assert sha(target)==sha(label) and np.array_equal(decoded,png) and model_hash(trainer.surrogate)==before
    proof=dict(state='passed',scope='one real training image; bounded learned coefficients; NOT formal export',
        image_id=Path(image_path).stem,source_label_sha256=sha(label),output_label_sha256=sha(target),
        native_parameter_schema_accepted=True,native_parameter_identity=identity,
        selected_parameter_shape=list(params['fourier_coeff'].shape),native_final_image_matches_export_exact=True,
        native_vs_export_max_abs_difference=max_abs_difference,comparison_tolerance=0.,
        comparison_context='original generation mode on both sides; stochastic training unchanged',
        decoded_PNG_roundtrip_exact=True,decoded_budget=audit,actual_composition_device=str(delta.device),
        TFSS_map_source=records[0],W_min=float(weight.min()),W_max=float(weight.max()),
        Teacher_unchanged=True,official_metrics_disabled=True,artifact_ingest_disabled=True,
        extra_model_forwards=0,formal_receipt_written=False)
    write_once(output/'export_link_proof.json',proof)
    return proof
