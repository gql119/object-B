"""Small lifecycle contracts; no torch import, dataset reads, or GPU."""
import ast,hashlib,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("recovery",ROOT/"scripts/recover_s4_tfss_full_victim.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class RecoveryContracts(unittest.TestCase):
 from contextlib import contextmanager
 @contextmanager
 def retained_case(self):
  root=ROOT/'execution'/'recovery_cpu_cases';root.mkdir(parents=True,exist_ok=True)
  yield tempfile.mkdtemp(dir=root) # All test artifacts retained; no cleanup.

 def test_actual_relative_bootstrap_is_excluded(self):
  argv=b'python\x00-B\x00scripts/recover_s4_tfss_full_victim.py\x00--config\x00CFG\x00--phase\x00bootstrap\x00'
  with patch.dict(r.os.environ,{'S4_FULL_RECOVERY_BOOTSTRAP':json.dumps(dict(pid=123,start_ticks=456,uid=1014))}),patch.object(r,'same',return_value=True),patch('pathlib.Path.read_bytes',return_value=argv):
   self.assertTrue(r.bound_bootstrap(123,{'_config_path':'CFG'}))
 def test_unrelated_pid_not_excluded(self):
  with patch.dict(r.os.environ,{'S4_FULL_RECOVERY_BOOTSTRAP':json.dumps(dict(pid=123,start_ticks=456,uid=1014))}):
   self.assertFalse(r.bound_bootstrap(789,{'_config_path':'CFG'}))
 def test_stale_bootstrap_identity_not_excluded(self):
  with patch.dict(r.os.environ,{'S4_FULL_RECOVERY_BOOTSTRAP':json.dumps(dict(pid=123,start_ticks=456,uid=1014))}),patch.object(r,'same',return_value=False):
   self.assertFalse(r.bound_bootstrap(123,{'_config_path':'CFG'}))

 def test_frozen_training_functions_and_all_original_python(self):
  base="1d10ad620ef01aad20e8c85d49465473d72c9a4e"
  for name in ["scripts/run_s4_tfss_full.py","ue_framework/s4_victim.py","ue_framework/s4_amp_observer.py"]:
   old=subprocess.check_output(["git","show",base+":"+name],cwd=ROOT)
   current=(ROOT/name).read_bytes().replace(b"\r\n",b"\n")
   self.assertEqual(old,current,name)
 def test_same_parent_output_rejected(self):
  c=dict(root="/allowed",output="/allowed/outputs/original",parent_output="/allowed/outputs/original")
  with self.assertRaises(ValueError):r.assert_config(c)
 def test_output_outside_boundary_rejected(self):
  c=dict(root="/allowed",output="/other/outputs/recovery",parent_output="/allowed/outputs/original")
  with self.assertRaises(ValueError):r.assert_config(c)
 def test_existing_parent_pid_blocks_recovery(self):
  with self.retained_case() as td:
   p=Path(td);(p/"pipeline_status.json").write_text(json.dumps(dict(run_id="old",source_commit="SCI",config_sha256="CFG")))
   c=dict(parent_run_id="old",science_commit="SCI",config_sha256="CFG",dead_parent_pids=[123])
   with patch.object(r,"assert_config",return_value=(p/"fresh",p,{})),patch.object(r,"identity",return_value={"pid":123}):
    with self.assertRaisesRegex(ValueError,"Prior bound PID"):r.validate_parent(c)
 def test_parent_hash_drift_blocks_recovery(self):
  with self.retained_case() as td:
   p=Path(td);(p/"pipeline_status.json").write_text(json.dumps(dict(run_id="old",source_commit="SCI",config_sha256="CFG")))
   (p/"immutable.json").write_text("changed")
   c=dict(parent_run_id="old",science_commit="SCI",config_sha256="CFG",dead_parent_pids=[123],parent_files={"immutable.json":"invalid"})
   with patch.object(r,"assert_config",return_value=(p/"fresh",p,{})),patch.object(r,"identity",return_value=None):
    with self.assertRaisesRegex(ValueError,"artifact changed"):r.validate_parent(c)
 def test_lifecycle_does_not_enable_noise_or_resume(self):
  tree=ast.parse((ROOT/"scripts/recover_s4_tfss_full_victim.py").read_text(encoding="utf-8"))
  calls={getattr(n.func,"id",getattr(n.func,"attr","")) for n in ast.walk(tree) if isinstance(n,ast.Call)}
  self.assertNotIn("optimize_s4_full",calls);self.assertNotIn("materialize_and_audit",calls)
  self.assertIn("train_evaluate",calls);self.assertIn("start_detached_supervisor",calls)
if __name__=="__main__":unittest.main(verbosity=2)

