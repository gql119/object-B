"""Record actual first native formal work, never infer it from launch alone."""
import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DOC=ROOT/'docs/S4_AT_ROBUSTNESS_20261027'
o=json.loads((HERE/'NEW_FORMAL_OBSERVATION.json').read_text());r=json.loads((HERE/'SELECTED_PROTOCOL_SCIENTIFIC_REVIEW.json').read_text())
assert o['healthy'] and o['progress']['actual_updates']>=1 and r['result']=='scientifically_correct' and r['decision']=='allow_run'
subprocess.run([sys.executable,'-B',str(HERE/'write_direct_report.py')],check=True)
p=DOC/'CURRENT_STATE.json';s=json.loads(p.read_text(encoding='utf8'))
s['speed_optimization'].update(state='FALLBACK_ORIGINAL_PROTOCOL_STARTED',decision='ORIGINAL_BATCH36_FRESH_FOUR_ARM_FALLBACK_NO_PROVEN_SPEEDUP',restart_executed=True,GPU_speedup_percent=None)
s['cuda_batch_optimization'].update(new_formal_run_started=True,decision='FALLBACK_ORIGINAL36_NO_NEW_BATCH_ADOPTED',scientific_ready=True,formal_ready=True)
s.update(formal_runs_started=2,formal_generations_with_actual_scientific_training=2,old_formal_aborted=True,
    preflight_sha256=hashlib.sha256((HERE/'SELECTED_ACCEPTED_PREFLIGHT.json').read_bytes()).hexdigest(),
    new_formal_run_spec_sha256=o['run_spec_sha256'],current_arm=o['current_arm'],AT_results=None,
    source_runtime_commit=o['source_commit'],last_monitor_evidence='execution/direct_stop_20261030/NEW_FORMAL_OBSERVATION.json',last_monitor_health='healthy')
p.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
marker='## NEW FRESH FOUR-ARM FORMAL CONTROLLER ACTUALLY TRAINING'
f=ROOT/'execution/AT_CONTINUE.md'
if marker not in f.read_text(encoding='utf8'):
    with f.open('a',encoding='utf8') as stream:
        stream.write('\n\n'+marker+' — '+o['UTC']+'\n'+
            'RunID '+o['run_id']+' RUNNING/healthy source '+o['source_commit']+'; OriginalBatch36 fallback after inconclusive performance, NO Candidate/newBatch adopted. Four fresh seed0 YAML200epoch arms same frozen PGD10/AT4step1/AT8step2/640/workers16/innerFP32/outerAMP/nativeSGD/nbs64/fullGT. Current '+o['current_arm']+' '+json.dumps(o['progress'])+'. Exact UID1014 PID-start identities '+json.dumps(o['identities'])+'. Model/EMA/optimizer finite and initialization matches actual current preflight. Same source native26, baseline7(diff0/tol0), phase61 exact PASS; single independent scientifically_correct/allow_run, mainclosure and all immutable gate upload/readbacks passed. Actual gate and RunSpec in SELECTED_RUNTIME.json and '+o['run_id']+'/.\n'+
            'Old20261029_v2 remains ABORTED/86complete, never resume/accept/relaunch; old locks/checkpoints/logs retained. New nondeleting exclusive serial queue was CPU production-control-smoked, only one formal controller/GPU child. Pending next3 arms run serially after actual prior arm acceptance. No more optimization probes, old CPU/CUDA/reviews or COCO repeats; 48/64/profile remain NOT RUN and excluded. Inspect new binding only with observe_new_formal.py, then after each accepted200epoch audit fullstate/source/init/continuousrows and reuse already computed4952/all20 lastprimary/bestsecondary. Keep SAME ACTIVE monitor/cadence until all4 accepted and final paired scores/provenance published. Archive HEAD will differ from immutable actual runtime; never hotpatch active release.\n')
print(json.dumps(dict(state=s['state'],run_id=o['run_id'],actual_updates=o['progress']['actual_updates'],old_aborted=True)))
