"""Lifecycle-only recovery: immutable completed Full data -> fresh unchanged victim.
No noise optimization, dataset export, numerical model/loss or optimizer override.
The prior output is read-only; every recovery artifact uses a new bound output.
"""
import argparse, csv, hashlib, json, os
from pathlib import Path
import subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_s4_robustness_pipeline import atomic,sha,identity,same,supervise,memory_scope

def read(p):return json.loads(Path(p).read_text())
def current():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def assert_config(c):
    root=Path(c['root'])
    out=Path(c['output']);parent=Path(c['parent_output'])
    if not out.is_relative_to(root/'outputs') or not parent.is_relative_to(root/'outputs') or out==parent:
        raise ValueError('Distinct authorized recovery output required')
    if sha(ROOT/c['lock'])!=c['config_sha256']:raise ValueError('Frozen Full protocol changed')
    if current()!=c['runtime_commit']:raise ValueError('Runtime commit differs')
    for rel,expected in c['scientific_files'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('Scientific function bytes changed: '+rel)
    return out,parent,read(ROOT/c['lock'])

def validate_parent(c, *, full_files=True):
    out,parent,lock=assert_config(c)
    st=read(parent/'pipeline_status.json')
    if (st['run_id'],st['source_commit'],st['config_sha256'])!=(c['parent_run_id'],c['science_commit'],c['config_sha256']):
        raise ValueError('Parent Full identity mismatch')
    if any(identity(pid) for pid in c['dead_parent_pids']):raise ValueError('Prior bound PID still exists; no recovery')
    for rel,expected in c['parent_files'].items():
        if sha(parent/rel)!=expected:raise ValueError('Completed Full artifact changed: '+rel)
    ns=read(parent/'noise/status.json');audit=read(parent/'dataset_integrity.json')
    if not(ns.get('schedule_complete') is True and ns['steps']==9144 and ns['epochs_completed']==24):
        raise ValueError('Full noise not complete')
    keys=['labels_byte_identical','clean_val_byte_identical','outside_target_zero','per_pixel_frozen_budget_passed','decoded_PNG_audited']
    if audit.get('state')!='passed' or audit.get('support_mode')!='bbox_full_target' or not all(audit.get(k) is True for k in keys):
        raise ValueError('Full pixel/label audit incomplete')
    if (audit['train'],audit['val'],audit['person_gt'],audit['changed_person_gt'])!=(16551,4952,13256,13256):
        raise ValueError('Full counts wrong')
    rows=read(parent/'materialization_manifest.json')
    if len(rows)!=16551 or len({r['id'] for r in rows})!=16551:raise ValueError('Manifest incomplete')
    if full_files:
        for r in rows:
            p=Path(r['output'])
            if not p.is_relative_to(parent/'dataset/images/train') or sha(p)!=r['output_sha256']:
                raise ValueError('Generated Full image mismatch')
            if sha(parent/'dataset/labels/train'/(r['id']+'.txt'))!=r['label_sha256']:raise ValueError('Generated Full label mismatch')
        clean=Path(c['clean_root'])
        for kind in ['images','labels']:
            paths=[p for p in (parent/'dataset'/kind/'val').iterdir() if p.is_file()]
            if len(paths)!=4952:raise ValueError('Clean validation count changed')
            for p in paths:
                if sha(p)!=sha(clean/kind/'val'/p.name):raise ValueError('Clean validation changed')
    return dict(state='passed',parent_run_id=c['parent_run_id'],parent_files=c['parent_files'],
        full_files_rehashed=full_files,train=16551,val=4952,noise_steps=9144,
        prior_victim_reused=False,exact_resume_claim=False,new_victim_seed=0,new_victim_epochs=200)

def link_completed(c):
    out,parent,_=assert_config(c)
    for rel in ['noise','dataset','clean_maps','source_audit.json','config_lock_snapshot.json',
                'dataset_integrity.json','materialization_manifest.json','SPATIAL_DIAGNOSTICS_SUMMARY.json']:
        dest=out/rel
        if dest.exists() or dest.is_symlink():raise FileExistsError('Recovery link collision: '+str(dest))
        dest.symlink_to(parent/rel,target_is_directory=(parent/rel).is_dir())

def worker(c):
    out,parent,lock=assert_config(c)
    # Same side effects that setup() established before original train_evaluate().
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    from ue_framework.s4_full_contract import load_full_config
    from scripts.run_s4_tfss_full import train_evaluate
    cfg=load_full_config(ROOT,lock)
    atomic(out/'worker_identity.json',dict(**identity(os.getpid()),memory_scope=memory_scope(identity(os.getpid())),
        science_commit=c['science_commit'],runtime_commit=c['runtime_commit'],recovery_run_id=c['run_id'],
        fresh_random_init=True,exact_resume=False,parent_output=str(parent),train_args=lock['victim_protocol']['train_args']))
    # This exact reviewed function reseeds Python/NumPy/CPU/CUDA and constructs a
    # random YAML model, native DetectionTrainer, native SGD/GradScaler and validators.
    train_evaluate(cfg,out,parent/'dataset',lock)

def smoke(c):
    out,_,_=assert_config(c)
    if out.exists():raise FileExistsError('Unique scope smoke output required')
    out.mkdir(parents=True)
    job=out/'jobs/synthetic';job.mkdir(parents=True)
    rc=supervise(job,[sys.executable,'-B','-c','import json,os;print(json.dumps(dict(pid=os.getpid(),CPU_only=True)))'],ROOT,True)
    child=read(job/'child.json');launch=read(job/'scope_launch.json')
    if rc or child['memory_scope']['path']==launch['supervisor_scope']['path']:raise ValueError('Ordinary scope isolation failed')
    atomic(out/'smoke_summary.json',dict(state='passed',runtime_commit=c['runtime_commit'],
        config_sha256=c['config_sha256'],scientific_files_exact=True,independent_ordinary_scope=True,
        OOM_policy_overrides=False,GPU_steps=0,real_dataset_reads=0,scope=child['memory_scope']['path']))

def controller(c):
    out,parent,lock=assert_config(c)
    if out.exists():raise FileExistsError('Fresh recovery only; no reset or resume')
    out.mkdir(parents=True)
    import fcntl
    with (out/'recovery.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(state='running',run_id=c['run_id'],source_commit=c['science_commit'],
            runtime_commit=c['runtime_commit'],config_sha256=c['config_sha256'],support_mode='bbox_full_target',
            stage='parent_full_data_verification',controller=identity(os.getpid()),
            generation_run_id=c['parent_run_id'],recovery=True,exact_victim_resume=False)
        def mark(stage,**kw):
            state.update(stage=stage,updated_at=time.time(),**kw);atomic(out/'pipeline_status.json',state);print(json.dumps(state),flush=True)
        try:
            mark('parent_full_data_verification')
            atomic(out/'PARENT_DATA_REUSE_AUDIT.json',validate_parent(c))
            link_completed(c)
            atomic(out/'RECOVERY_PROVENANCE.json',dict(**c,authorization='2026-10-22 direct human diagnose/fix/continue authorization',
                finished_generation_reused=True,no_noise_regeneration=True,fresh_victim200=True,exact_resume=False))
            from scripts.run_s4_tfss_full import resource_admission
            while True:
                mark('waiting_victim_capacity')
                try:capacity=resource_admission()
                except RuntimeError as error:
                    if not str(error).startswith('Capacity/I-O admission pending; no launch:'):raise
                    mark('waiting_victim_capacity',resource_wait=str(error));time.sleep(900);continue
                atomic(out/'resource_admission.json',dict(samples=capacity,passed=True));break
            # Read-only guard against another owned actual training workload.
            for p in Path('/proc').iterdir():
                if not p.name.isdigit() or int(p.name)==os.getpid():continue
                try:
                    if p.stat().st_uid!=os.getuid():continue
                    args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
                    if args and 'python' in Path(args[0]).name:
                        entries=[Path(a).name for a in args[1:] if a.endswith('.py')]
                        if any(a.startswith(('train','run_s4','run_s5','recover_s4')) for a in entries):
                            raise ValueError('Another owned training root active: '+str(p))
                except OSError:pass
            mark('victim_training_and_evaluation')
            job=out/'jobs/victim';job.mkdir(parents=True)
            rc=supervise(job,[sys.executable,'-B',str(Path(__file__).resolve()),'--config',c['_config_path'],'--phase','worker'],ROOT,True)
            if rc:raise RuntimeError('Isolated victim exited '+str(rc)+'; retained exit.json/scope_result.json; no automatic batch resume')
            mark('reporting')
            from scripts.s4_full_report import finalize
            finalize(out,ROOT,c['science_commit'],lock)
            mark('complete',state='complete',completed_at_unix=time.time())
        except BaseException as error:mark(state['stage'],state='failed',error=repr(error));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--phase',choices=['controller','worker','scope-smoke'],required=True)
    a=p.parse_args();c=read(a.config);c['_config_path']=str(Path(a.config).resolve())
    out=Path(c['output'])
    for key,name in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch')]:
        folder=out/'runtime'/name
        # controller/smoke requires absent output; only worker creates runtime dirs.
        if a.phase=='worker':folder.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(folder)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
    {'controller':controller,'worker':worker,'scope-smoke':smoke}[a.phase](c)
if __name__=='__main__':main()

