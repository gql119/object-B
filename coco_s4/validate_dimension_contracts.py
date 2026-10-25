"""Fail-closed selected-coordinate and frozen-config contracts, no GPU/data."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import yaml

from .runtime import ROOT,now,write_once
from .dimension_protocol import PROTOCOL,validate_receipt_dimension
from .phase_fourier import validate_phase_config,validate_fourier_source_config


def run(output):
    cfg=ROOT/'docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml'
    before=yaml.safe_load(subprocess.check_output(['git','-C',str(ROOT),'show',
                     '38ada45fa4942ff64693f736b6c7863c8e1a3fb9:'+cfg.relative_to(ROOT).as_posix()]))
    actual=yaml.safe_load(cfg.read_text(encoding='utf-8'))
    validate_fourier_source_config(actual)
    candidate=copy.deepcopy(actual)
    assert candidate['methods']['legacy_kproto_ret']['semantic']['safe_bg'].pop('dimension_protocol')==PROTOCOL
    assert candidate==before,'A non-dimension scientific setting changed'
    valid=dict(dimension_protocol=PROTOCOL,total_feature_channels=256,eligible_channel_count=78,
               selected_channels=list(range(39)),selected_channel_count=39)
    assert validate_receipt_dimension(valid)==39
    rejected=[]
    for field,value in (('selected_channel_count',24),('eligible_channel_count',12),('eligible_channel_count',True),
                        ('total_feature_channels',255.5),('dimension_protocol','old24'),
                        ('selected_channels',[float(i) for i in range(39)]),
                        ('selected_channels',list(range(38))+[37]),('selected_channels',list(reversed(range(39)))),
                        ('selected_channels',list(range(256,295)))):
        bad=copy.deepcopy(valid);bad[field]=value
        try:validate_receipt_dimension(bad)
        except (ValueError,TypeError):rejected.append(field)
        else:raise AssertionError('Invalid selected-space receipt accepted: '+field)
    config=json.loads((ROOT/'coco_s4/configs/fourier24.json').read_text())
    validate_phase_config(config)
    for key,value in (('selected_space_dimension',39),('dimension_protocol','legacy24'),('epochs',23),
                      ('workflow_run_id','s4_coco_experiment_a_seed0_20261025_v3')):
        bad=dict(config);bad[key]=value
        try:validate_phase_config(bad)
        except ValueError:rejected.append(key)
        else:raise AssertionError('Invalid Fourier contract accepted: '+key)
    parsed=[]
    for name in ('dimension_protocol.py','dimension_rebuild.py','dimension_release.py',
                 'phase_contract.py','phase_fourier.py','phase_export.py','phase_victim.py','phase_final_report.py'):
        ast.parse((ROOT/'coco_s4'/name).read_text(encoding='utf-8'),filename=name);parsed.append(name)
    report=dict(state='passed',UTC=now(),invalid_contracts_rejected=len(rejected),AST_modules=parsed,
                entire_scientific_config_equal_except_explicit_dimension_protocol=True,
                GPU_units=0,model_forwards=0,formal_registration=False)
    write_once(Path(output),report);print(json.dumps(report));return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)
