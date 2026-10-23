"""Freeze dataset-dependent schedule size without changing original S4 epochs."""
def expected_noise_schedule(num_classes,eligible_images):
    if num_classes==20:return {'epochs':24,'optimizer_steps':9144,'batch_size':16,'eot_samples':2}
    if num_classes!=80 or eligible_images<=0:raise ValueError('Invalid COCO optimization manifest')
    return {'epochs':24,'optimizer_steps':24*((eligible_images+15)//16),'batch_size':16,'eot_samples':2}

def admit_gpu(predecessor,queue,checks):
    if predecessor.get('pipeline_run_id')!=queue['predecessor_run_id']:return 'BLOCKED_PREDECESSOR_IDENTITY'
    if predecessor.get('active_job') is not None:return 'WAITING_PREVIOUS_PIPELINE'
    if any(predecessor.get('stages',{}).get(k)!='ACCEPTED' for k in ('cutout','cutmix','mixup')):return 'WAITING_PREVIOUS_PIPELINE'
    if predecessor.get('state')!='BLOCKED_PROTOCOL_APPROVAL' or not checks.get('first3_archive_verified'):return 'WAITING_PREVIOUS_ARCHIVE'
    required=['snapshot_published','official_split_verified','all_images_verified','data_audit_passed','CPU_passed','source_clean','queue_conflict_free','no_live_own_training_root','capacity_fresh','current_GPU_smoke_passed','single_scientific_review_closed']
    if not all(checks.get(k) is True for k in required):return 'BLOCKED_NEW_EXPERIMENT_GATES'
    return 'ADMITTED'
