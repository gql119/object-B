"""Freeze four scientific RunSpecs locally; does not launch RRCTL or a server waiter."""
import argparse
from .common import ROOT, PROTOCOL, protocol, run_config, sha, commit, write_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', default=None)
    args = parser.parse_args()
    source = args.source_commit or commit()
    p = protocol()
    for run_id in p['order']:
        cfg = run_config(run_id)
        out = p['remote_owned_root'] + '/outputs/' + run_id
        spec = dict(schema='s4.at.scientific-runspec.v1', **cfg, source_commit=source,
                    primary='last.pt@200', secondary='best.pt', train_args=p['train_args'],
                    native_sources=p['native_sources'], train_count=16551, val_count=4952,
                    source_S4_artifacts=p['reused_artifacts'], output_root=out,
                    current_state='PROTOCOL_LOCKED_PENDING_CUDA_REVIEW_AND_COCO_COMPLETION',
                    command=['/project/guoqinglong/miniconda3/envs/cfbdm/bin/python', '-B', '-m',
                             'robustness_s4_at.worker', '--formal', '--run-id', run_id, '--device', '0',
                             '--out', out, '--source-commit', source, '--preflight',
                             p['remote_owned_root'] + '/accepted_preflight.json'],
                    progress_path='audit/progress.json', cleanup_policy='retain_all',
                    official_run_started=False, same_budget_Clean_required=True)
        write_once(ROOT / 'robustness_s4_at/runspecs' / (run_id + '.json'), spec)
    print('Four protocol RunSpecs frozen; no RRCTL launch or production READY created')


if __name__ == '__main__':
    main()
