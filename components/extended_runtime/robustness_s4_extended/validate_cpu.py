"""Retained CPU evidence only. Never calls a GPU, model trainer or real dataset scan."""
import ast,contextlib,datetime,hashlib,importlib.metadata,io,json,os,sys,time,unittest
from pathlib import Path
from .common import ROOT,CONFIG,load,sha,atomic

def main():
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
    for folder in ['cpu_yolo','cpu_mpl']:(ROOT/'execution'/folder).mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'execution/cpu_yolo');os.environ['MPLCONFIGDIR']=str(ROOT/'execution/cpu_mpl')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    files=sorted((ROOT/'robustness_s4_extended').rglob('*.py'))
    for p in files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    start=time.monotonic();buffer=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'robustness_s4_extended/tests'),pattern='test_*.py',top_level_dir=str(ROOT))
    with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):r=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    e=ROOT/'execution';e.mkdir(exist_ok=True);(e/'CPU_TEST_RAW.txt').write_text(buffer.getvalue(),encoding='utf-8')
    result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASSED' if r.wasSuccessful() else 'FAILED',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),elapsed_seconds=time.monotonic()-start,parsed_files=len(files),GPU_steps=0,actual_detection_dataset_CPU=True,real_VOC_images_read=0,synthetic_fixtures_retained=True,config_sha256=sha(CONFIG),packages={k:importlib.metadata.version(k) for k in ['torch','ultralytics','numpy','Pillow']},runtime_limitations='Local CPU builds/NumPy differ from locked remote; GPU reachability and one scientific review pending; no official AP',raw_sha256=sha(e/'CPU_TEST_RAW.txt'))
    atomic(ROOT/'robustness_s4_extended/reports/CPU_TEST_RESULT.json',result)
    print(json.dumps(result));print(buffer.getvalue()[-6500:])
    if not r.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
