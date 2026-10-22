import time,unittest
from pathlib import Path
from robustness_s4_extended.common import ROOT,atomic,sha
from robustness_s4_extended.selection import select_and_lock,load_decision,SelectionBlocked

class DecisionRecoveryTests(unittest.TestCase):
    def fixture(self):
        root=ROOT/'execution/decision_recovery_cases'/str(time.time_ns())
        old=root/'old';new=root/'new';full=root/'full';parent=root/'parent'
        for d in [old,new,full,parent]:d.mkdir(parents=True)
        atomic(full/'pipeline_status.json',dict(state='complete'))
        atomic(new/'PREDECESSOR_ACCEPTANCE.json',dict(state='COMPLETED_AND_ACCEPTED',run_id='FULL',native_state_sha256=sha(full/'pipeline_status.json')))
        atomic(parent/'materialization_manifest.json',[]);atomic(parent/'dataset_integrity.json',{})
        b=dict(poisoned_dataset_path=str(parent/'dataset'),parent_output=str(parent),parent_materialization_manifest_sha256=sha(parent/'materialization_manifest.json'),parent_dataset_integrity_sha256=sha(parent/'dataset_integrity.json'),parent_noise_run_id='S4',parent_noise_checkpoint_sha256='NOISE')
        d=dict(batch_id='OLD',selection_source_commit='OLD_SOURCE',selection_config_sha256='OLD_CONFIG',decision_rule_sha256='RULE',validation_status='VALIDATED_AND_LOCKED',selected_method='S4_ORIGINAL',selected_poison_dataset=b['poisoned_dataset_path'],selected_dataset_manifest=str(parent/'materialization_manifest.json'),selected_dataset_sha256=b['parent_materialization_manifest_sha256'],selected_integrity_sha256=b['parent_dataset_integrity_sha256'])
        d['selected_dataset_manifest']=b['parent_output']+'/materialization_manifest.json'
        atomic(old/'S4_FULL_DECISION.json',d)
        for name in ['S4_FULL_COMPARISON.csv','S4_FULL_COMPARISON.md']:(old/name).write_text('old immutable report')
        origin=dict(output_root=str(old),batch_id='OLD',source_commit='OLD_SOURCE',config_sha256='OLD_CONFIG',decision_sha256=sha(old/'S4_FULL_DECISION.json'),full_acceptance_sha256=sha(new/'PREDECESSOR_ACCEPTANCE.json'))
        c=dict(output_root=str(new),pipeline_run_id='NEW',predecessor=dict(run_id='FULL',output=str(full)),baseline=b,selection=dict(rule_sha256='RULE',locked_decision_reuse=origin))
        return c,old,new
    def test_exact_decision_reused_without_selection(self):
        c,old,new=self.fixture();self.assertEqual(select_and_lock(c)['selected_method'],'S4_ORIGINAL')
        self.assertEqual((old/'S4_FULL_DECISION.json').read_bytes(),(new/'S4_FULL_DECISION.json').read_bytes())
        self.assertEqual(load_decision(c)['batch_id'],'OLD')
    def test_mutated_previous_decision_blocks(self):
        c,old,new=self.fixture();(old/'S4_FULL_DECISION.json').write_text('{}')
        with self.assertRaises(SelectionBlocked):select_and_lock(c)
        self.assertFalse((new/'S4_FULL_DECISION.json').exists())
    def test_changed_acceptance_blocks(self):
        c,old,new=self.fixture();atomic(new/'PREDECESSOR_ACCEPTANCE.json',dict(state='COMPLETED_AND_ACCEPTED',run_id='FULL',native_state_sha256='OTHER'))
        with self.assertRaises(SelectionBlocked):select_and_lock(c)
