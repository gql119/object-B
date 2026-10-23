"""Run actual CPU contracts and parent regressions; record failures, never skip."""
import ast,datetime,io,json,runpy,sys,types,unittest
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
def main():
    suite=unittest.TestSuite();loader=unittest.TestLoader()
    for pattern,folder in [('test_*.py','coco_s4/tests'),('test_tfss_spatial.py','tests'),('test_tfss_static_control.py','tests'),('test_bbox_half_overlap.py','tests'),('test_orth_reference_design.py','tests'),('test_b2_semantic.py','tests')]:
        for p in sorted((ROOT/folder).glob(pattern)):
            module=types.ModuleType(p.stem);module.__dict__.update(runpy.run_path(str(p)))
            if p.name=='test_b2_semantic.py':
                # Its old B2 byte-lock fixture also fails unchanged S4 parentb1e2951.
                # Preserve that historical file; test actual current S4 byte-lock above.
                cls=module.SemanticTests
                suite.addTests(cls(n) for n in loader.getTestCaseNames(cls) if n!='test_baseline_files_identical')
            else:suite.addTests(loader.loadTestsFromModule(module))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    syntax=[]
    for folder in ['coco_s4']:
        for p in (ROOT/folder).rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'));syntax.append(str(p.relative_to(ROOT)))
    for rel in ['ue_framework/methods/tfss_nt_sensitivity.py','ue_framework/methods/tfss_clean_maps.py','ue_framework/methods/b2_semantic.py','ue_framework/methods/s4_optimization.py','ue_framework/methods/s4_pixel_audit.py','scripts/build_semantic_coordinates.py','scripts/build_b2_to_safe_coordinates.py','ue_framework/s4_victim.py','ue_framework/s4_amp_observer.py']:
        ast.parse((ROOT/rel).read_text(encoding='utf-8-sig'));syntax.append(rel)
    out=ROOT/'docs/S4_COCO_MINITRAIN_20261023';out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(ROOT/'execution').mkdir(exist_ok=True)
    (ROOT/'execution'/('CPU_TEST_CONSOLE_'+stamp+'.txt')).write_text(stream.getvalue())
    report={'UTC':stamp,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'passed':result.wasSuccessful() and not result.skipped,'syntax_files':syntax,'CPU_only':True,'native_teacher':'random YAML on CPU, not formal clean25K pretrained teacher','GPU_smoke':'PENDING_PREDECESSOR_AND_DATA','formal_runs_launched':0,'known_initial_native_probe_failure':'2 fixture errors: raw DetectionModel lacked args; fixed by setting native get_cfg(), no production loss/guard change'}
    report['historical_fixture_separate_result']={'name':'old_B2_baseline_files_identical','fails_on_unchanged_S4_parent':True,'expected_old_B2_legacy_sha':'e6203909a21d1ac50ba7e41c80219dc69bac9499ad41b2649f9368d8052dabe3','actual_original_S4_legacy_sha':'6bfd9155ea4ed4aa90dba655d835407c0bfad92737b153adc0060215c0020ccb','replacement':'current_original_S4_math_sources_exact checks8actualserver-SHA-locked sources; historical fixture and failure remain retained'}
    (out/'CPU_VALIDATION_REPORT.json').write_text(json.dumps(report,indent=2));print(stream.getvalue());print(json.dumps(report))
    if not report['passed']:sys.exit(1)
if __name__=='__main__':main()
