import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scheduler',ROOT/'scripts/run_s4_robustness_pipeline.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Contracts(unittest.TestCase):
    def test_failures_never_turn_numerical_error_into_environment_retry(self):
        self.assertEqual(m.classify('AssertionError: non-finite raw loss plus CUDA out of memory'),'BLOCKED')
        self.assertEqual(m.classify('CUDA out of memory'),'RETRYABLE_FAILURE')
        self.assertEqual(m.classify('unexpected undefined internal failure'),'FAILED')
    def test_atomic_state_and_exact_queue(self):
        tmp=tempfile.mkdtemp(prefix='pipeline-contract-',dir=ROOT/'execution')
        p=Path(tmp)/'state.json';m.atomic(p,{'state':'RUNNING'});m.atomic(p,{'state':'ACCEPTED'})
        self.assertEqual(m.load(p),{'state':'ACCEPTED'})
        self.assertFalse(p.with_name('state.json.writing').exists())
        self.assertEqual(m.ORDER,['existing','gray','bdr','phase_A','gauss','meanf','medianf','data_aug','phase_B','archive'])
    def test_frozen_protocol_is_not_changed_by_orchestrator(self):
        c=m.load(ROOT/'docs/S4_ROBUSTNESS_PIPELINE_20261020/PIPELINE_CONFIG.json')
        self.assertFalse(c['AT_authorized']);self.assertFalse(c['reboot_automatic_restart'])
        self.assertEqual(c['science_commit'],'e17c24c54aae4cdae4bcc4703facf2a57052d2c8')
        for p,h in c['science_sha256'].items():self.assertEqual(m.sha(ROOT/p),h,p)
        self.assertNotIn('resume',m.load(ROOT/'docs/S4_ROBUSTNESS_20261019/S4_ROBUSTNESS_LOCK.json')['victim_protocol']['train_args'])
    def test_scope_isolation_has_no_resource_protection_override(self):
        unit,argv=m.scope_argv(ROOT/'execution/test-scope',['python','frozen_train.py','--batch','36'],ROOT)
        self.assertTrue(unit.startswith('s4rb-job-'))
        self.assertEqual(argv[:4],['systemd-run','--user','--scope','--quiet'])
        self.assertEqual(argv[-4:],['python','frozen_train.py','--batch','36'])
        self.assertFalse(any(x.startswith(('--property','-p','--slice')) for x in argv))
        self.assertEqual(m.classify('Result=oom-kill'),'RETRYABLE_FAILURE')
        self.assertEqual(m.classify('AssertionError with Result=oom-kill'),'BLOCKED')

if __name__=='__main__':unittest.main()
