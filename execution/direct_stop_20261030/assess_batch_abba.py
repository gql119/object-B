"""Assess new-batch throughput, never claim cross-batch trajectory equality.

Thermal/load limits and paired uncertainty match the prelocked engineering
assessor. Different native accumulation/BN/warmup/update counts are disclosed,
not edited to make trajectories match. This script does not launch a run.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path


def assess(paths):
    rows=[json.loads(p.read_text()) for p in paths];reasons=[]
    def require(ok,why):
        if not ok:reasons.append(why)
    require(len(rows)==4,'EXACTLY_FOUR_ABBA_SAMPLES_REQUIRED')
    if len(rows)!=4:return dict(state='INCONCLUSIVE',reasons=reasons,gain=None)
    batches=[r['native_batch'] for r in rows]
    require(batches[0]==batches[3]==36 and batches[1]==batches[2] and batches[1] in (48,64,72),'WRONG_FROZEN_BATCH_ABBA_ORDER')
    fields=('source_commit','entry_sha256','implementation','arm','epsilon_pixels','imgsz','workers','AMP','fixture_identity_sha256','pool_mode','initial_state')
    for field in fields:require(all(r.get(field)==rows[0].get(field) for r in rows),'UNMATCHED_'+field)
    for a,b in ((rows[0],rows[3]),(rows[1],rows[2])):
        for field in ('stable_entry_state','actual_outer_batches','actual_SGD','actual_AMP_skips','augmented_GT_sequence','observed_native_shapes','final_model','final_optimizer','final_scaler','final_EMA'):
            require(a.get(field)==b.get(field),'WITHIN_BATCH_UNMATCHED_'+field)
    temperatures=[];clocks=[];cpu=[];fractions=[];speeds=[]
    for r in rows:
        require(r['state']=='PASS_FINITE_NATIVE_BENCHMARK_SAMPLE' and r['CUDA_performance_measured'] and r['CUDA_initialized'],'NOT_PASSED_REAL_CUDA')
        require((r['imgsz'],r['workers'],r['AMP'],r['pool_mode'])==(640,16,True,'common720'),'NOT_NATIVE_COMMON_POOL')
        require(r['actual_selected_train_count']==720 and r['full_native_PGD10_and_original_guards'],'INCOMPLETE_POOL_OR_GUARDS')
        require(r['actual_weight_decay']==[0.0005625000000000001],'ACTUAL_DECAY_CHANGED')
        require(not r['timing']['profile_instrumented'],'INSTRUMENTED_SPEED_SAMPLE')
        measured=r['measured_rows'];roi=[x for x in r['resource_telemetry'] if measured[0]['start']<=x['monotonic']<=measured[-1]['end']]
        require(len(roi)>=3,'INSUFFICIENT_ROI_SAMPLES')
        busy=[];thermal=[];power=[]
        for x in roi:
            require(not x.get('sampling_error') and not x.get('other_GPU_pids'),'CONTAMINATED_GPU_SAMPLE')
            require(x.get('RAM_available_GiB',0)>=32,'RAM_GATE_DROPPED')
            g=x['gpu'];temperatures.append(float(g['temperature.gpu']));clocks.append(float(g['clocks.sm']))
            states=[g[k] for k in ('clocks_throttle_reasons.sw_thermal_slowdown','clocks_throttle_reasons.hw_thermal_slowdown','clocks_throttle_reasons.sw_power_cap')]
            require(all(s in ('Active','Not Active') for s in states),'UNREADABLE_THROTTLE_STATE')
            thermal.append(any(s=='Active' for s in states[:2]));power.append(states[2]=='Active')
            if 'host_CPU_busy_percent' in x:busy.append(x['host_CPU_busy_percent'])
        require(bool(busy),'MISSING_HOST_CPU_CONTROL')
        if busy:cpu.append(statistics.mean(busy))
        if thermal:fractions.append((statistics.mean(thermal),statistics.mean(power)))
        speed=r['timing']['images_per_second'];require(math.isfinite(speed) and speed>0,'INVALID_THROUGHPUT');speeds.append(speed)
    require(bool(temperatures) and max(temperatures)-min(temperatures)<=2,'THERMAL_RANGE_EXCEEDS_PRELOCKED_2C')
    require(bool(clocks) and min(clocks)>0 and max(clocks)/min(clocks)-1<=.05,'CLOCK_VARIATION_EXCEEDS_PRELOCKED_5PERCENT')
    if cpu:require(max(cpu)-min(cpu)<=10,'UNCONTROLLED_HOST_CPU_LOAD')
    if fractions:
        require(max(x[0] for x in fractions)-min(x[0] for x in fractions)<=.25,'UNCONTROLLED_THERMAL_THROTTLE_FRACTION')
        require(max(x[1] for x in fractions)-min(x[1] for x in fractions)<=.25,'UNCONTROLLED_POWER_CAP_FRACTION')
    result=dict(state='INCONCLUSIVE',reasons=sorted(set(reasons)),new_training_protocol=True,cross_batch_trajectory_equivalent=False,
        batches=batches,images_per_second=speeds,gain=None,lower95_gain=None,scientific_ready=False,formal_ready=False,
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        protocols=[{k:r.get(k) for k in ('native_batch','native_batches_per_epoch','native_warmup_batches','native_post_warmup_accumulate','nominal_weight_decay','actual_weight_decay','actual_SGD','actual_AMP_skips','cuda_peak_allocated_MiB','epoch_checkpoint_costs')} for r in rows])
    if not reasons:
        logs=[math.log(speeds[1]/speeds[0]),math.log(speeds[2]/speeds[3])];mean=statistics.mean(logs)
        lower=mean-12.706204736*statistics.stdev(logs)/math.sqrt(2)
        result.update(state='MEASURED_NEW_BATCH_ABBA',gain=math.exp(mean)-1,lower95_gain=math.exp(lower)-1,
            preferred_throughput_gate_passed=math.exp(lower)-1>=.15)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summaries',type=Path,nargs=4,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();v=assess(a.summaries)
    with a.out.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
    print(json.dumps(v))
