"""Aggregate only four independently accepted last@200 AT arms."""
import csv
from pathlib import Path
from .common import load, protocol, sha, write_once
from .metrics import pair
from .data import VOC_NAMES


def aggregate(root):
    root, p = Path(root), protocol()
    rows, per_class, summary = [], [], {}
    accepted = {}
    for run_id in p['order']:
        out = root / run_id
        receipt, metrics = load(out / 'accepted.json'), load(out / 'last_primary/metrics.json')
        if receipt['state'] != 'ACCEPTED' or receipt['epochs'] != 200 \
                or sha(out / 'last_primary/metrics.json') != receipt['primary_metrics_sha256']:
            raise ValueError('AT report cannot ingest an unaccepted/changed arm')
        accepted[run_id] = receipt
        nt = [i for i in range(20) if i != 14]
        rows.append(dict(run_id=run_id, Person_AP50=metrics['AP50'][14],
                         NT_AP50=sum(metrics['AP50'][i] for i in nt) / 19,
                         all_AP50=sum(metrics['AP50']) / 20,
                         Person_AP50_95=metrics['AP50_95'][14],
                         NT_AP50_95=sum(metrics['AP50_95'][i] for i in nt) / 19,
                         NT_AP75=sum(metrics['AP75'][i] for i in nt) / 19,
                         last_sha256=receipt['last_sha256']))
        per_class.extend(dict(run_id=run_id, class_id=i, name=VOC_NAMES[i],
                              **{k: metrics[k][i] for k in ('AP50', 'AP75', 'AP50_95', 'Recall', 'Precision')})
                         for i in range(20))
    for eps in (4, 8):
        left, right = f'S4AT{eps}_Clean_seed0', f'S4AT{eps}_Poison_seed0'
        a, b = accepted[left], accepted[right]
        for field in ('source_commit', 'protocol_sha256', 'attack', 'model_yaml_sha256', 'initial_model_sha256'):
            if a[field] != b[field]:
                raise ValueError('Mismatched Clean/Poison AT pair: ' + field)
        summary[str(eps)] = pair(load(root / left / 'last_primary/metrics.json'),
                                 load(root / right / 'last_primary/metrics.json'))
        c, s = [next(row for row in rows if row['run_id'] == name) for name in (left, right)]
        ref = p['reference_ST_exact']
        summary[str(eps)].update(Person_recovery_vs_S4_ST_pp=100 * (s['Person_AP50'] - ref['poison_Person_AP50']),
            Clean_AT_vs_Clean_ST_Person_pp=100 * (c['Person_AP50'] - ref['clean_Person_AP50']),
            Clean_AT_vs_Clean_ST_NT_pp=100 * (c['NT_AP50'] - ref['clean_NT_AP50']),
            ST_reference_precision='exact archived accepted ST fractions; independent reference only')
    for name, values in (('AT_MAIN_RESULTS.csv', rows), ('AT_PER_CLASS_RESULTS.csv', per_class)):
        with (root / name).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    write_once(root / 'AT_PAIRED_SCORES.json', summary)
    write_once(root / 'ST_REFERENCE_ONLY.json', p['reference_ST_exact'])
    return summary
