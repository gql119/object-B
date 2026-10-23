"""Synthetic selection/lineage contracts; no actual Full evaluation or GPU."""
import copy,json,time,unittest
from pathlib import Path
from unittest.mock import patch
from robustness_s4_extended.common import ROOT,atomic,sha
from robustness_s4_extended import selection as s
from robustness_s4_extended.predecessor import generation_output

RULE=dict(decision_rule_version='conservative_pareto_S_20261022_v1',S_improvement_tolerance=1e-9,other_guard_tolerance=0)
def metric(person=.04,nt=.7):
 d={k+'_per_class':[nt]*20 for k in ['AP50','AP75','AP50_95','Recall','Precision']}
 d['AP50_per_class'][14]=person
 return d
class SelectionTests(unittest.TestCase):
 def setUp(self):self.clean=metric(.8,.8);self.old=metric()
 def compare(self,m,**kw):return s.compare_metrics(self.clean,self.old,m,RULE,full_validated=kw.get('valid',True),protocols_equal=kw.get('same',True))
 def test_full_advantage(self):self.assertEqual(self.compare(metric(.03,.71))['selected_method'],'S4_TFSS_FULL')
 def test_equal_full_defaults_original(self):self.assertEqual(self.compare(metric())['selected_method'],'S4_ORIGINAL')
 def test_lower_person_cannot_hide_NT_damage(self):self.assertEqual(self.compare(metric(.001,.65))['selected_method'],'S4_ORIGINAL')
 def test_WCD_guard_despite_higher_mean(self):
  m=metric(.03,.75);m['AP50_per_class'][0]=.3
  self.assertEqual(self.compare(m)['selected_method'],'S4_ORIGINAL')
 def test_higher_person_cannot_win(self):self.assertEqual(self.compare(metric(.05,.75))['selected_method'],'S4_ORIGINAL')
 def test_failed_full_is_blocked_not_fallback(self):
  with self.assertRaises(s.SelectionBlocked):self.compare(metric(),valid=False)
 def test_protocol_mismatch_blocked(self):
  with self.assertRaises(s.SelectionBlocked):self.compare(metric(),same=False)
 def test_missing_class_cannot_select(self):
  m=metric();m['Recall_per_class'].pop()
  with self.assertRaises(ValueError):self.compare(m)
 def test_undefined_denominator_blocked(self):
  with self.assertRaises(s.SelectionBlocked):s.compare_metrics(metric(0,0),self.old,metric(),RULE,full_validated=True,protocols_equal=True)
 def test_rule_tuning_blocked(self):
  r=dict(RULE,S_improvement_tolerance=0)
  with self.assertRaises(s.SelectionBlocked):s.compare_metrics(self.clean,self.old,metric(.03,.71),r,full_validated=True,protocols_equal=True)
 def test_signed_scores_not_clipped(self):
  d=self.compare(metric(.9,.9));self.assertLess(d['comparison_metrics']['S4_TFSS_FULL']['P'],0);self.assertGreater(d['comparison_metrics']['S4_TFSS_FULL']['U'],1)
 def test_unknown_method_cannot_point_to_arbitrary_source(self):
  with self.assertRaises(s.SelectionBlocked):s.descriptor({},'unknown')
 def test_existing_decision_is_reused_not_reselected(self):
  root=ROOT/'execution/selection_cases'/str(time.time_ns());root.mkdir(parents=True)
  (root/'S4_FULL_DECISION.json').write_text('{}')
  with patch.object(s,'load_decision',return_value={'locked':True}) as read:
   self.assertEqual(s.select_and_lock({'output_root':str(root)}),{'locked':True});read.assert_called_once()
 def test_mixed_dataset_decision_rejected(self):
  root=ROOT/'execution/selection_cases'/str(time.time_ns());root.mkdir(parents=True)
  c=dict(output_root=str(root),pipeline_run_id='B',selection={'rule_sha256':'RULE'},baseline=dict(poisoned_dataset_path='/original',parent_output='/origparent',parent_materialization_manifest_sha256='MAN',parent_dataset_integrity_sha256='INT',parent_noise_run_id='original',parent_noise_checkpoint_sha256='NOISE'))
  p=root/'S4_FULL_DECISION.json'
  atomic(p,dict(batch_id='B',selection_source_commit='C',selection_config_sha256='CFG',decision_rule_sha256='RULE',validation_status='VALIDATED_AND_LOCKED',selected_method='S4_ORIGINAL',selected_poison_dataset='/full',selected_dataset_manifest='/origparent/materialization_manifest.json',selected_dataset_sha256='MAN',selected_integrity_sha256='INT'))
  with patch.object(s,'commit',return_value='C'),patch.object(s,'sha',return_value='CFG'):
   with self.assertRaisesRegex(s.SelectionBlocked,'data changed or mixed'):s.load_decision(c,check_files=False)

class RecoveryLineageTests(unittest.TestCase):
 def test_normal_generation_unchanged(self):
  p=Path('/unchanged');self.assertEqual(generation_output({},p,{}),p)
 def test_wrong_generation_boundary_rejected(self):
  with self.assertRaisesRegex(ValueError,'boundary'):
   generation_output({'recovery':dict(generation_output='/else/outputs/g',full_root='/allowed',generation_run_id='g')},Path('/allowed/outputs/r'),{})
 def test_wrong_recovery_runtime_rejected(self):
  parent=ROOT/'execution/selection_cases'/str(time.time_ns())/'outputs/g';out=parent.parent/'r';out.mkdir(parents=True)
  atomic(out/'RECOVERY_PROVENANCE.json',dict(run_id='r',runtime_commit='bad',science_commit='SCI',config_sha256='CFG',parent_run_id='g',parent_output=str(parent)))
  atomic(out/'PARENT_DATA_REUSE_AUDIT.json',dict(state='passed'))
  pre=dict(run_id='r',source_commit='SCI',config_sha256='CFG',recovery=dict(generation_output=str(parent),full_root=str(parent.parent.parent),generation_run_id='g',runtime_commit='good'))
  with self.assertRaisesRegex(ValueError,'provenance'):generation_output(pre,out,{})
if __name__=='__main__':unittest.main(verbosity=2)

