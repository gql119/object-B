"""CPU/std-library guard and schema checks, explicitly not GPU evidence."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from bounded_common import ROOT
from robustness_s4_at.common import commit,sha,write_once
from assess_benchmarks import assess
from cuda_resume import adapted_run_source
from prepare_specs import build,cases


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    files=('bounded_common.py','cuda_equivalence.py','cuda_benchmark.py','cuda_resume.py','gpu_telemetry.py','assess_benchmarks.py','prepare_specs.py')
    hashes={}
    for name in files:
        path=Path(__file__).parent/name;ast.parse(path.read_text(encoding='utf8'));hashes[name]=sha(path)
    source,sites=adapted_run_source();compile(source,'<resume adapter CPU compile>','exec')
    required=("_expected_phase", "_expected_steps", "_expected_resumed_updates", "actual_outer_batches", "tolerance=0")
    if not all(s in source for s in required):raise ValueError('Original resume cadence/tolerance was not retained')
    from remote_run_control.models import RunSpec
    template=json.loads((ROOT/'execution/s4_at_cuda_native_20261029_gpu03.json').read_text())
    binding=json.loads((Path(__file__).parent/'RETAINED_GPU_FIXTURE_BINDING.json').read_text())
    tags=[]
    for case in cases():
        value=build(template,binding,commit(),case);RunSpec.from_dict(value);tags.append(value['run_id'])
        if '--device' not in value['workload']['argv'] or '0' not in value['workload']['argv']:raise ValueError('No explicit CUDA device')
        if value['metadata']['max_total_native_outer_batches']>100:raise ValueError('Probe native budget exceeds100')
    if len(tags)!=len(set(tags)):raise ValueError('Duplicate future RunID')
    base=dict(state='PASS_FINITE_NATIVE_BENCHMARK_SAMPLE',simulation_only=True,source_commit='SIMULATION_ONLY',entry_sha256='SIMULATION_ONLY',
        arm='clean',epsilon_pixels=4,native_batch=36,imgsz=640,workers=16,AMP=True,fixture_identity_sha256='SIMULATION_ONLY',
        pool_mode='common720',initial_state={'model':'fixed'},stable_entry_state={'model':'fixed'},actual_outer_batches=76,
        actual_SGD=39,actual_AMP_skips=6,augmented_GT_sequence=['fixed'],observed_native_shapes=[[36,3,640,640]],
        final_model='same',final_optimizer='same',final_scaler='same',final_EMA='same',
        CUDA_performance_measured=True,CUDA_initialized=True,full_native_PGD10_and_original_guards=True,
        measured_rows=[{'start':0,'end':1},{'start':9,'end':10}],
        timing={'images_per_second':20.0,'profile_instrumented':False},resource_telemetry=[])
    for n in (2,4,6,8):
        base['resource_telemetry'].append(dict(monotonic=n,other_GPU_pids=[],RAM_available_GiB=70,
            host_CPU_busy_percent=40,gpu={'temperature.gpu':'85','clocks.sm':'1800',
            'clocks_throttle_reasons.sw_thermal_slowdown':'Active',
            'clocks_throttle_reasons.hw_thermal_slowdown':'Not Active','clocks_throttle_reasons.sw_power_cap':'Active'}))
    positive=[]
    for i,implementation in enumerate(('original','03','03','original')):
        r=deepcopy(base);r['implementation']=implementation
        if i in (1,2):r['timing']['images_per_second']=24.0
        positive.append(r)
    def evaluate(label,rows):
        folder=out/label;folder.mkdir();paths=[]
        for i,row in enumerate(rows):
            path=folder/(str(i)+'.json');write_once(path,row);paths.append(path)
        result=assess(paths);write_once(folder/'assessment.json',result)
        if result['restart_allowed']:raise ValueError('A measurement assessor must never authorize retirement/restart')
        return result
    results=[]
    result=evaluate('SIMULATED_MATCHED',positive)
    if result['state']!='SIMULATION_ONLY_GUARD_CALCULATION' or result['Old_New_GPU_gain'] is not None:raise ValueError('Positive guard fixture failed or fabricated a GPU gain')
    results.append(dict(case='SIMULATED_MATCHED',accepted_by_guard=True,actual_GPU_evidence=False))
    def negative(label,change):
        rows=deepcopy(positive);change(rows);r=evaluate(label,rows)
        if r['state']!='INCONCLUSIVE':raise ValueError('Guard accepted: '+label)
        results.append(dict(case=label,rejected=True,reasons=r['reasons']))
    negative('CPU_not_CUDA',lambda r:r[1].update(CUDA_performance_measured=False))
    negative('wrong_order',lambda r:r[0].update(implementation='03'))
    negative('different_TAL_GT',lambda r:r[1].update(augmented_GT_sequence=['different']))
    negative('different_model_update',lambda r:r[1].update(final_model='different'))
    negative('extra_AMP_skips',lambda r:r[1].update(actual_AMP_skips=7))
    negative('profile_instrumented',lambda r:r[1]['timing'].update(profile_instrumented=True))
    negative('GPU_competitor',lambda r:r[1]['resource_telemetry'][0].update(other_GPU_pids=[99999]))
    negative('thermal_drift',lambda r:r[1]['resource_telemetry'][0]['gpu'].update({'temperature.gpu':'88'}))
    negative('clock_drift',lambda r:r[1]['resource_telemetry'][0]['gpu'].update({'clocks.sm':'1500'}))
    negative('CPU_load_drift',lambda r:r[1]['resource_telemetry'][0].update(host_CPU_busy_percent=90))
    negative('missing_telemetry',lambda r:r[1].update(resource_telemetry=[]))
    if 'torch' in sys.modules:raise ValueError('Std-library contract test unexpectedly imported Torch')
    report=dict(state='PASS_CPU_CONTRACT_AND_SCHEMA_ONLY',simulation_only=True,actual_GPU_jobs=0,
        actual_model_forwards=0,actual_throughput_measurements=0,torch_imported=False,CUDA_initialized=False,
        source_files=hashes,resume_AST_sites=sites,real_RRCTL_schema_checked=len(tags),unique_RunIDs=len(tags),
        numerical_tolerance_changed=False,actual_RunSpecs_prepared=False,actual_GPU_readiness=False,
        synthetic_guard_cases=results)
    write_once(out/'summary.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
