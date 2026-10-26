"""Validate the original native selected-frequency artifact without inventing fields."""
def validate_native_params(params, *, num_classes, target_class_id, feature_layer, identity=None):
    import torch
    if num_classes != 80 or target_class_id != 0 or feature_layer != 'model.21':
        raise ValueError('Artifact consumer must be the bound COCO80 Person0 P5 model')
    if (params.get('method') != 'legacy_kproto_ret' or params.get('support_mode') != 'bbox_half_overlap'
            or params.get('target_class_id') != target_class_id or params.get('feature_layer') != feature_layer
            or abs(float(params.get('eps',0))-8/255) > 1e-9
            or ('num_classes' in params and params['num_classes'] != num_classes)):
        raise ValueError('Original S4 native parameter protocol differs')
    coords=params.get('coords');coeff=params.get('fourier_coeff');small=params.get('suppress_small')
    if (not isinstance(coords,list) or len(coords)!=16 or len({tuple(c) for c in coords})!=16
            or any(len(c)!=2 or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in c) for c in coords)
            or not torch.is_tensor(coeff) or tuple(coeff.shape)!=(16,3) or not torch.isfinite(coeff).all()
            or not torch.is_tensor(small) or not torch.isfinite(small).all()):
        raise ValueError('Native selected16 parameter tensors/coordinates are invalid')
    saved=params.get('s4_identity')
    if not isinstance(saved,dict) or (identity is not None and saved!=identity):
        raise ValueError('Native original S4 source/data/checkpoint identity is missing or changed')
    return saved
