"""Targeted F01 selection regression, using the actual helper on CPU or CUDA.

Explicit empty-geometry fixtures are not native TAL positives or real labels.
The independent float64 oracle uses the actual float32 endpoints, not their
unrepresentable decimal constructor literals.
"""
def run(device_name):
    import math,torch
    from types import SimpleNamespace
    from .risk_fallback import risk_candidate_mask
    device=torch.device(device_name)
    assigner=SimpleNamespace(select_candidates_in_gts=lambda anchors,box,valid:
        torch.zeros((1,1,len(anchors)),dtype=torch.bool,device=anchors.device))
    anchors=torch.tensor([[4.,4.],[12.,4.]],dtype=torch.float32,device=device)
    box=torch.tensor([[[7.99,3.99,8.010001,4.01]]],dtype=torch.float32,device=device)
    mask,selection=risk_candidate_mask(assigner,anchors,box)
    endpoints=box[0,0].double().cpu().tolist();points=anchors.double().cpu().tolist()
    size=[endpoints[i+2]-endpoints[i] for i in (0,1)]
    center=[math.fsum((endpoints[i],endpoints[i+2]))/2 for i in (0,1)]
    distances=[sum(((p[i]-center[i])/size[i])**2 for i in (0,1)) for p in points]
    oracle=min(range(len(points)),key=lambda i:(distances[i],i))
    old_center=(box[0,0,:2]+box[0,0,2:])/2;old_size=box[0,0,2:]-box[0,0,:2]
    old_distances=((anchors.double()-old_center.double())/old_size.double()).square().sum(-1)
    assert int(old_distances.argmin())==0 and oracle==1 and mask.cpu().tolist()==[False,True]
    assert selection['indices']==[1] and 'nearest' in selection['rule']
    tie_box=torch.tensor([[[7.5,3.5,8.5,4.5]]],dtype=torch.float32,device=device)
    tie,tie_selection=risk_candidate_mask(assigner,anchors,tie_box)
    assert tie.cpu().tolist()==[True,False] and tie_selection['indices']==[0]
    invalid=0
    for bad in (torch.tensor([[[1.,1.,1.,2.]]],device=device),torch.tensor([[[2.,1.,1.,2.]]],device=device),torch.full((1,1,4),float('nan'),device=device)):
        try:risk_candidate_mask(assigner,anchors,bad)
        except ValueError:invalid+=1
        else:raise AssertionError('Invalid GT accepted')
    assert invalid==3
    # An existing geometric selection must never be recomputed by the repair.
    native_mask=torch.tensor([True,False],device=device)
    geometry=SimpleNamespace(select_candidates_in_gts=lambda anchors,box,valid:native_mask[None,None])
    unchanged,detail=risk_candidate_mask(geometry,anchors,box)
    assert torch.equal(unchanged,native_mask) and detail['indices']==[0] and 'native GT geometric' in detail['rule']
    return dict(state='passed',device=str(device),cuda_index=anchors.device.index,
        actual_helper='coco_s4.risk_fallback.risk_candidate_mask',fixture_scope='explicit empty-mask arithmetic fixture; no real label mutation',
        actual_float32_endpoints=endpoints,independent_float64_center=center,independent_float64_distances=distances,
        old_float32_derived_distances=old_distances.cpu().tolist(),old_selected_index=0,correct_selected_index=1,
        actual_selection=selection,exact_center_tie_lowest_native_index=True,invalid_boxes_rejected=invalid,
        existing_geometric_mask_unchanged=True,model_forwards=0,parameter_updates=0,official_metrics_disabled=True,
        artifact_ingest_disabled=True,formal_READY_written=False)
