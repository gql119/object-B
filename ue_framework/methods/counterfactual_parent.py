"""Exact 1pp parent selection and full-run eligibility, no metric defaults."""
import math


REQUIRED_GATES = (
    'noise_complete','materialization_complete','label_integrity','pixel_budget',
    'support_domain','victim_200','last_checkpoint','best_checkpoint',
    'voc20_last_evaluation','voc20_best_evaluation','metrics_complete','identity_archive',
)
PROTOCOL_KEYS = ('dataset_sha256','clean_val_sha256','architecture','seed',
                 'epochs','initialization','validator_sha256','evaluation_role')


def select_parent(s0,s1):
    for evidence in (s0,s1):
        if any(evidence['gates'].get(k) is not True for k in REQUIRED_GATES):
            raise ValueError('Parent selection requires complete formal experiment/archive')
        if any(k not in evidence['protocol'] for k in PROTOCOL_KEYS):
            raise ValueError('Incomplete evaluation protocol')
        if (evidence['protocol']['evaluation_role'] != 'last.pt' or evidence['protocol']['epochs'] != 200
                or evidence['protocol']['initialization'] != 'random'):
            raise ValueError('Parent requires random-init 200-epoch last.pt protocol')
        for key in ('person_ap50','nt_map50'):
            v=evidence['metrics'][key]
            if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
                raise ValueError('Invalid AP50; expect fraction, not percent')
        if not evidence['run_id'] or not evidence['identity']['budget_rule']:
            raise ValueError('Missing RunID/budget provenance')
        for key,length in (('git_sha',40),('config_sha256',64),('last_checkpoint_sha256',64)):
            value=evidence['identity'][key]
            if len(value)!=length or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Missing or malformed archive identity: '+key)
    if any(s0['protocol'][k] != s1['protocol'][k] for k in PROTOCOL_KEYS):
        raise ValueError('S0/S1 evaluation protocols differ')
    # AP values stored as decimal strings retain the intended exact 1pp boundary.
    from decimal import Decimal
    dt=Decimal(str(s1['metrics']['person_ap50']))-Decimal(str(s0['metrics']['person_ap50']))
    dn=Decimal(str(s1['metrics']['nt_map50']))-Decimal(str(s0['metrics']['nt_map50']))
    accepted=dt<=Decimal('.01') and dn>=Decimal('-.01')
    return dict(status='LOCKED',parent='S1' if accepted else 'S0',
                delta_target=float(dt),delta_nt=float(dn),passes_tolerance=accepted,
                rule='delta_target <= 0.01 AND delta_nt >= -0.01',
                tolerance_kind='preregistered engineering 1pp; not statistical significance',
                s0=s0,s1=s1,selected_identity=(s1 if accepted else s0)['identity'],
                victim_checkpoint_role='evaluation provenance only, never learner initialization')


def require_formal_lock(lock):
    """DRAFT/UNSET never falls through to formal training defaults."""
    required=('parent_selection_sha256','parent_source_sha','parent_config_sha256',
              'reference_checkpoint_sha256','budget_identity','dataset_identity',
              'cnlp_config','cup_config','noise_schedule','victim_protocol',
              'train_query_manifest_sha256','calibration_sha256','admission_evidence_sha256')
    if lock.get('status') != 'LOCKED' or lock.get('formal_generation_enabled') is not True:
        raise ValueError('Formal S2/S3 disabled until Parent/calibration/admission locks')
    if any(k not in lock or lock[k] is None for k in required):
        raise ValueError('Incomplete formal configuration lock')
    if lock.get('method') not in ('S2-CNLP','S3-CNLP-CUP'):
        raise ValueError('Unexpected formal method')
    from .counterfactual_utility import CNLPConfig,CUPConfig
    cfg=lock['cnlp_config']
    CNLPConfig(cfg['inner_steps'],cfg['inner_lr'],cfg['lambda_cnlp'],matching_policy=cfg.get('matching_policy','strict_native_all_GT'))
    if cfg['lambda_cnlp']<=0:
        raise ValueError('Formal CNLP requires a positive train-calibrated lambda')
    for key in ('parent_selection_sha256','parent_config_sha256','reference_checkpoint_sha256',
                'train_query_manifest_sha256','calibration_sha256','admission_evidence_sha256'):
        value=lock[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid formal identity: '+key)
    source=lock['parent_source_sha']
    if not isinstance(source,str) or len(source)!=40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('Invalid Parent source SHA')
    if not lock['noise_schedule'] or not lock['dataset_identity'] or not lock['budget_identity']:
        raise ValueError('Missing schedule/dataset/budget bindings')
    victim=lock['victim_protocol']
    if victim.get('epochs')!=200 or victim.get('initialization')!='random' or victim.get('primary')!='last.pt':
        raise ValueError('Formal victim must inherit random-init200/last.pt primary')
    if lock['method']=='S3-CNLP-CUP' and not lock['cup_config']:
        raise ValueError('S3 requires a separately audited CUP config')
    if lock['method']=='S3-CNLP-CUP':
        CUPConfig(**lock['cup_config'])
    if lock['method']=='S2-CNLP' and lock['cup_config'] != {}:
        raise ValueError('S2 must have no CUP weighting')


def validate_paired_locks(s2,s3):
    require_formal_lock(s2); require_formal_lock(s3)
    if s2['method'] != 'S2-CNLP' or s3['method'] != 'S3-CNLP-CUP':
        raise ValueError('Expected ordered S2/S3 pair')
    different={'method','cup_config','run_id','output_root'}
    if {k:v for k,v in s2.items() if k not in different} != {k:v for k,v in s3.items() if k not in different}:
        raise ValueError('S2/S3 differ beyond CUP aggregation and run/output identity')
    if not s2.get('run_id') or not s3.get('run_id') or s2['run_id']==s3['run_id']:
        raise ValueError('Independent S2/S3 RunIDs required')
    if not s2.get('output_root') or not s3.get('output_root') or s2['output_root']==s3['output_root']:
        raise ValueError('Independent S2/S3 output roots required')
