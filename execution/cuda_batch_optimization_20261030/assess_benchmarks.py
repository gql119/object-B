"""Read retained actual ABBA samples. Never schedules, selects or restarts runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

THERMAL_RANGE_C=2.0
CLOCK_RELATIVE_RANGE=.05
HOST_CPU_MEAN_RANGE_PERCENT=10.0
MIN_ROI_TELEMETRY_SAMPLES=3
THROTTLE_FRACTION_RANGE=.25


def assess(paths):
    runs=[json.loads(p.read_text(encoding='utf8')) for p in paths]
    reasons=[]
    def require(condition,reason):
        if not condition:reasons.append(reason)
    require(len(runs)==4,'EXACTLY_FOUR_PREDECLARED_ABBA_SAMPLES_REQUIRED')
    if len(runs)!=4:return dict(state='INCONCLUSIVE',reasons=reasons,GPU_gain=None)
    ids=[r.get('implementation') for r in runs]
    require(ids[0]=='original' and ids[3]=='original' and ids[1] in ('02','03') and ids[1]==ids[2],'ORDER_MUST_BE_ORIGINAL_CANDIDATE_CANDIDATE_ORIGINAL')
    invariant=('source_commit','entry_sha256','arm','epsilon_pixels','native_batch','imgsz','workers','AMP',
        'fixture_identity_sha256','pool_mode','initial_state','stable_entry_state','actual_outer_batches',
        'actual_SGD','actual_AMP_skips','augmented_GT_sequence','observed_native_shapes',
        'final_model','final_optimizer','final_scaler','final_EMA')
    for field in invariant:require(all(r.get(field)==runs[0].get(field) for r in runs), 'UNMATCHED_'+field)
    for r in runs:
        require(r.get('state')=='PASS_FINITE_NATIVE_BENCHMARK_SAMPLE','NATIVE_SAMPLE_DID_NOT_PASS')
        require(r.get('CUDA_performance_measured') is True and r.get('CUDA_initialized') is True,'NOT_ACTUAL_CUDA')
        require(r.get('native_batch')==36 and r.get('pool_mode')=='common720','ENGINEERING_ABBA_REQUIRES_COMMON_POOL_BATCH36')
        require(r.get('imgsz')==640 and r.get('workers')==16 and r.get('AMP') is True,'NOT_FROZEN_NATIVE_SHAPE')
        require(r.get('full_native_PGD10_and_original_guards') is True,'GUARD_EVIDENCE_MISSING')
        require(not r.get('timing',{}).get('profile_instrumented'),'INSTRUMENTED_SAMPLE_CANNOT_PROVE_SPEEDUP')
    samples=[];cpu_means=[];temperatures=[];clocks=[];thermal=[];power=[];fractions=[]
    for r in runs:
        rows=r.get('measured_rows',[])
        if not rows:require(False,'MEASUREMENT_ROWS_MISSING');continue
        roi=[x for x in r.get('resource_telemetry',[]) if rows[0]['start']<=x['monotonic']<=rows[-1]['end']]
        require(len(roi)>=MIN_ROI_TELEMETRY_SAMPLES,'INSUFFICIENT_THERMAL_SAMPLING')
        cpu=[x['host_CPU_busy_percent'] for x in roi if 'host_CPU_busy_percent' in x]
        require(bool(cpu),'MISSING_HOST_CPU_CONTROL')
        if cpu:cpu_means.append(statistics.mean(cpu))
        run_thermal=[];run_power=[]
        for x in roi:
            require(not x.get('sampling_error') and not x.get('other_GPU_pids'),'RESOURCE_ERROR_OR_COMPETING_GPU')
            require(x.get('RAM_available_GiB',0)>=32,'RAM_GATE_DROPPED_DURING_SAMPLE')
            try:
                g=x['gpu'];temperatures.append(float(g['temperature.gpu']));clocks.append(float(g['clocks.sm']))
                thermal.append((g['clocks_throttle_reasons.sw_thermal_slowdown'],g['clocks_throttle_reasons.hw_thermal_slowdown']))
                power.append(g['clocks_throttle_reasons.sw_power_cap'])
                run_thermal.append(any(v=='Active' for v in thermal[-1]));run_power.append(power[-1]=='Active')
            except (KeyError,ValueError):require(False,'UNREADABLE_GPU_THERMAL_CLOCK_FIELDS')
        if run_thermal:fractions.append((statistics.mean(run_thermal),statistics.mean(run_power)))
        timing=r.get('timing',{});value=timing.get('images_per_second')
        require(isinstance(value,(int,float)) and math.isfinite(value) and value>0,'INVALID_ACTUAL_THROUGHPUT')
        samples.append(value)
    if temperatures:require(max(temperatures)-min(temperatures)<=THERMAL_RANGE_C,'THERMAL_RANGE_EXCEEDS_PRELOCKED_2C')
    if clocks:require(min(clocks)>0 and max(clocks)/min(clocks)-1<=CLOCK_RELATIVE_RANGE,'CLOCK_VARIATION_EXCEEDS_PRELOCKED_5PERCENT')
    require(bool(thermal) and all(v in ('Active','Not Active') for pair in thermal for v in pair),'UNREADABLE_THERMAL_THROTTLE_STATE')
    require(bool(power) and all(v in ('Active','Not Active') for v in power),'UNREADABLE_POWER_CAP_STATE')
    if fractions:
        require(max(v[0] for v in fractions)-min(v[0] for v in fractions)<=THROTTLE_FRACTION_RANGE,'UNCONTROLLED_THERMAL_THROTTLE_FRACTION')
        require(max(v[1] for v in fractions)-min(v[1] for v in fractions)<=THROTTLE_FRACTION_RANGE,'UNCONTROLLED_POWER_CAP_FRACTION')
    if cpu_means:require(max(cpu_means)-min(cpu_means)<=HOST_CPU_MEAN_RANGE_PERCENT,'UNCONTROLLED_HOST_CPU_LOAD')
    result=dict(schema='s4.at.engineering-abba-assessment.v1',state='INCONCLUSIVE',reasons=sorted(set(reasons)),
        simulation_only=any(r.get('simulation_only',False) for r in runs),
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        prelocked_thermal_range_C=THERMAL_RANGE_C,prelocked_relative_clock_range=CLOCK_RELATIVE_RANGE,
        prelocked_host_CPU_range_percent=HOST_CPU_MEAN_RANGE_PERCENT,minimum_ROI_samples=MIN_ROI_TELEMETRY_SAMPLES,
        prelocked_throttle_fraction_range=THROTTLE_FRACTION_RANGE,observed_throttle_fractions=fractions,
        Old_New_GPU_gain=None,lower95_GPU_gain=None,net_total_time_saving=None,
        restart_allowed=False,scientific_review_complete=False,production_control_retirement_proved=False)
    if reasons:return result
    # Two temporally paired comparisons: A0/B1 and A3/B2. df1 Student-t gives a
    # deliberately wide uncertainty interval; point estimate alone is no gate.
    logs=[math.log(samples[1]/samples[0]),math.log(samples[2]/samples[3])]
    mean=statistics.mean(logs);sd=statistics.stdev(logs)
    lower=mean-12.706204736*sd/math.sqrt(2)
    result.update(state='MEASURED_MATCHED_ABBA',Old_New_GPU_gain=math.exp(mean)-1,
        lower95_GPU_gain=math.exp(lower)-1,
        measured_throughput_gate_passed=math.exp(lower)-1>=.15,
        note='Only same-batch engineering evidence. New batches, four-arm correctness/resume, one review, net time and safe retirement remain independent gates.')
    if result['simulation_only']:
        result.update(state='SIMULATION_ONLY_GUARD_CALCULATION',
            synthetic_math_test=dict(gain=result['Old_New_GPU_gain'],lower95=result['lower95_GPU_gain']),
            Old_New_GPU_gain=None,lower95_GPU_gain=None,measured_throughput_gate_passed=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summaries',type=Path,nargs=4,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();value=assess(a.summaries)
    with a.out.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps(value))
