"""Analyze retained TRAIN32 layer diagnostics, without model or data forwards."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return mean(values) if values else None


def analyze(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Analysis output must be new; preserve previous evidence')
    summary = json.loads((source/'layer_probe_summary.json').read_text())
    protocol = json.loads((Path(__file__).resolve().parents[1]/'configs/s5_layer_probe.json').read_text())
    if (summary['state'] != 'complete' or summary['optimizer_updates'] != 30
            or summary['CUP_active'] or not summary['official_metrics_disabled']
            or summary['config_hash'] != hashlib.sha256(
                (Path(__file__).resolve().parents[1]/'configs/s5_layer_probe.json').read_bytes()).hexdigest()):
        raise ValueError('Completed fixed Stage B identity required')
    traces = [json.loads(line) for line in (source/'trace.jsonl').read_text().splitlines() if line]
    objects = json.loads((source/'retention_objects.json').read_text())
    classes = json.loads((source/'retention_classes.json').read_text())
    gt = json.loads((source/'GT_detection_proxy.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'P3_P5_GRADIENT_COMPARISON.csv', traces)
    write_csv(output/'P3_P5_CLASS_GRADIENTS.csv', classes)

    grouped = defaultdict(list)
    for row in objects:
        grouped[(row['batch'], row['role'], row['step'], row['post_five_updates'],
                 row['layer'], row['cls'], row['size'])].append(row)
    retention = []
    for key, rows in sorted(grouped.items()):
        batch, role, step, post, layer, cls, size = key
        active = [r for r in rows if r['included']]
        retention.append(dict(batch=batch, role=role, step=step, post_five_updates=post,
            layer=layer, cls=cls, size=size, observations=len(rows),
            included_observations=len(active), ineligible=sum(not r['eligible_image'] for r in rows),
            invalid_pool=sum(not r['pool_valid'] for r in rows),
            mean_squared_L2_sum=average(active, 'raw_loss'),
            mean_squared_L2_per_channel=mean([r['raw_loss']/r['channels'] for r in active]) if active else None,
            channels=rows[0]['channels'],
            count_unit='ROI/EOT observations, not independent images'))
    write_csv(output/'P3_P5_NON_TARGET_RETENTION.csv', retention)

    gains = next(r['GT_native_box_cls_dfl_gains'] for r in traces
                 if r.get('GT_native_box_cls_dfl_gains') is not None)
    gt_rows = []
    for row in gt:
        a, b = row['clean_components'], row['poison_components']
        delta = [y-x for x, y in zip(a, b)]
        gt_rows.append({k:v for k,v in row.items() if k not in ('clean_components','poison_components')} |
            dict(clean_box=a[0], clean_cls19=a[1], clean_DFL=a[2],
                 poison_box=b[0], poison_cls19=b[1], poison_DFL=b[2],
                 delta_box=delta[0], delta_cls19=delta[1], delta_DFL=delta[2],
                 signed_native_GT_loss_delta=sum(x*y for x,y in zip(delta,gains))))
    write_csv(output/'P3_P5_GT_DETECTION_PROXY.csv', gt_rows)
    final_classes = defaultdict(list)
    for row in gt_rows:
        if row['post_five_updates']:
            final_classes[(row['batch'],row['role'],row['cls'])].append(row)
    class_comparison = []
    for key, rows in sorted(final_classes.items()):
        class_comparison.append(dict(batch=key[0], role=key[1], cls=key[2],
            object_EOT_observations=len(rows),
            signed_native_GT_loss_delta=average(rows,'signed_native_GT_loss_delta'),
            delta_box=average(rows,'delta_box'), delta_cls19=average(rows,'delta_cls19'),
            delta_DFL=average(rows,'delta_DFL')))
    write_csv(output/'P3_P5_FINAL_CLASS_PROXY.csv', class_comparison)

    comparisons = []
    for batch in (0,1):
        final = {role:[r for r in traces if r['batch']==batch and r['role']==role
                       and r['post_five_updates']] for role in ('P3_base','P5_base')}
        row = dict(batch=batch)
        for role, layer in (('P3_base','P3'),('P5_base','P5')):
            rows = final[role]
            if len(rows) != 2:
                raise ValueError('Missing actual final-state EOT pair')
            for name in ('protection_proxy','GT_detection_signed_classmean_delta','actual_RMS',
                         'P5_missing_original_ROIs'):
                row[role+'_'+name] = average(rows,name)
            for name in ('protect_cosine','elementwise_conflict_fraction',
                         'opposing_protect_projection_ratio','weighted_ret_grad_norm',
                         'ret_to_protect_norm_ratio'):
                row[role+'_'+name] = average(rows,layer+'_'+name)
        comparisons.append(row)
    write_csv(output/'P3_P5_FINAL_BATCH_COMPARISON.csv', comparisons)
    valid = (summary['original_S4_vs_P3_noop_max_difference'] <= protocol['equivalence_absolute_tolerance']
             and len(summary['trials']) == 6
             and all(t['optimizer_updates']==5 and t['frozen_model_unchanged']
                     and t['adaptive_model_unchanged'] for t in summary['trials'])
             and all(r['finite_gradient'] for r in traces))
    evidence = dict(source_commit=summary['source_commit'], source=str(source.resolve()),
        layer_probe_complete=True, baseline_and_trial_gates=valid,
        original_S4_vs_P3_noop_max_difference=summary['original_S4_vs_P3_noop_max_difference'],
        maximum_missing_original_P3_ROIs=max(r.get('P5_missing_original_ROIs',0) for r in traces),
        runtime_trials=summary['trials'], final_batch_comparisons=comparisons,
        counts=dict(trace=len(traces), ROI_observations=len(objects), class_observations=len(classes),
                    GT_object_observations=len(gt_rows)),
        layer_decision='PENDING_MAIN_AGENT_PROTOCOL_ASSESSMENT',
        limitations=['Two fixed train batches; EOT/ROI observations are dependent.',
                     'Native GT proxies are same-model diagnostics, not victim AP.',
                     'Raw cross-layer feature losses cannot rank layer quality.',
                     'No official metric, CUP update, or formal S5 result is computed.'])
    (output/'P3_P5_ANALYSIS.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(evidence,indent=2,allow_nan=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    analyze(args.source,args.output)
