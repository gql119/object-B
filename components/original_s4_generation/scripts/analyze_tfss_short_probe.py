"""Output-only CSV/figure preparation. Never load a model or launch training."""
import argparse
import csv
import json
from pathlib import Path


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix+key
        if isinstance(item, dict):
            result.update(flatten(item, name+'.'))
        elif isinstance(item, list):
            result[name] = json.dumps(item)
        else:
            result[name] = item
    return result


def table(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('probe_root')
    a = p.parse_args()
    source = Path(a.probe_root)
    status = json.loads((source/'status.json').read_text())
    if status['state'] != 'complete':
        raise ValueError('Retain partial evidence; do not label an incomplete probe complete')
    out = source/'analysis';out.mkdir(exist_ok=False)
    curves, images, nt = [], [], []
    for trial in status['trials']:
        name = f"batch{trial['batch']}_{trial['method']}"
        for row in json.loads((source/name/'trace.json').read_text()):
            curves.append(flatten({k:v for k,v in row.items() if k != 'per_image'}))
            for per_image in row['per_image']:
                index = per_image['image_index']
                identity = {'method': row['method'], 'batch': row['batch'], 'step': row['step'],
                            'eot': row['eot'], 'image_id': row['fixed_image_ids'][index]}
                images.append(dict(identity, **flatten(per_image)))
        for row in json.loads((source/name/'nt_gt_diagnostics.json').read_text()):
            clean = {(r['gt_index'], r['cls']):r for r in row['clean']}
            for poison in row['poison']:
                key = poison['gt_index'], poison['cls']
                if key not in clean:
                    raise ValueError('GT identity mismatch in detection proxy')
                item = {'method': trial['method'], 'batch': trial['batch'], 'image_id': row['image_id'],
                        'gt_index': key[0], 'cls': key[1], 'forward': row['forward']}
                for term in ['box', 'cls_bce19', 'dfl', 'weighted_loss', 'positive_count']:
                    item['clean_'+term] = clean[key][term]
                    item['poison_'+term] = poison[term]
                    item['delta_'+term] = poison[term]-clean[key][term]
                nt.append(item)
    table(out/'BATCH_STEP_CURVES.csv', curves)
    table(out/'PER_IMAGE_GRADIENT_STRENGTH.csv', images)
    table(out/'NON_TARGET_GT_CLASSWISE.csv', nt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for loss in ['L_BG', 'L_rel', 'L_ret', 'L_total']:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), squeeze=False)
        for batch in [0, 1]:
            for method in sorted({r['method'] for r in curves}):
                rows = [r for r in curves if r['batch'] == batch and r['method'] == method]
                if not rows:
                    continue
                steps = sorted({r['step'] for r in rows})
                axes[0, batch].plot(steps, [np.mean([r[loss] for r in rows if r['step'] == step]) for step in steps], label=method)
            axes[0, batch].set(title=f'batch {batch}: {loss}', xlabel='optimizer step (pre-update forward)')
        axes[0, 1].legend(fontsize=6);fig.tight_layout()
        fig.savefig(out/(loss+'.png'), dpi=160);plt.close(fig)
    # Fixed representative indices chosen before outcomes; no source RGB export.
    for batch, indices in [(0, [0, 1]), (1, [0, 1, 8, 15])]:
        path = source/f'maps_batch{batch}.npz'
        if not path.exists():
            continue
        with np.load(path) as maps:
            for index in indices:
                fig, axes = plt.subplots(1, 2, figsize=(7, 3))
                for ax, key, title in zip(axes, ['q', 'nt_risk'], ['Qt (invalid regions preserved, see validity mask)', 'NT risk']):
                    ax.imshow(maps[f'{key}_{index}'], vmin=0, vmax=1, cmap='viridis');ax.set_title(title, fontsize=7);ax.axis('off')
                fig.tight_layout();fig.savefig(out/f'heatmap_b{batch}_i{index}.png', dpi=160);plt.close(fig)
    (out/'ANALYSIS_STATUS.json').write_text(json.dumps({'status':'DESCRIPTIVE_OUTPUTS_READY',
        'phase':status['phase'], 'candidate_decision':'INSUFFICIENT_EVIDENCE',
        'required_next':'Check A/B/C and class/image stability against locked protocol before recommendation; no formal launch'}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
