"""Run the bounded local CPU contracts and save actual evidence; no GPU launch."""
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
runtime=ROOT/"execution/local_yolo";runtime.mkdir(parents=True,exist_ok=True)
os.environ["YOLO_CONFIG_DIR"]=str(runtime)
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np
import torch
torch.set_num_threads(1)
from ue_framework.s4_full_contract import sha

def main():
    doc=ROOT/"docs/S4_TFSS_FULL_20261021"
    files=["ue_framework/methods/s4_tfss_full.py","ue_framework/methods/s4_full_optimization.py",
           "ue_framework/s4_full_contract.py","scripts/run_s4_tfss_full.py","scripts/s4_full_report.py","scripts/evaluate_s4_tfss_full.py",
           "scripts/check_s4_tfss_full_cpu.py","tests/test_s4_tfss_full.py"]
    for path in files:compile((ROOT/path).read_text(),path,"exec")
    started=time.perf_counter()
    loader=unittest.TestLoader()
    suite=unittest.TestSuite([loader.discover(str(ROOT/"tests"),pattern=pattern)
                             for pattern in ("test_s4_tfss_full.py","test_tfss_spatial.py")])
    stream=io.StringIO()
    outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    log=stream.getvalue();(doc/"CPU_TEST_LOG.txt").write_text(log,encoding="utf-8")
    probe=sys.modules["test_s4_tfss_full"]
    _,pairs=probe.FullContracts().compose_pair([probe.P,probe.DOG,probe.HORSE],training=True)
    old,new=pairs[0][1][1].detach(),pairs[1][1][1].detach()
    target,other=probe.domains((32,39,3),[probe.P,probe.DOG,probe.HORSE])
    overlap=torch.from_numpy(target&other)
    image=np.full((32,39,3),128,dtype=np.uint8)
    png=probe.export_project(image,np.ones_like(image,dtype=np.float32),[probe.P,probe.DOG],8/255)
    delta=png.astype(np.int16)-image.astype(np.int16)
    report=dict(schema="s4.tfss-full.cpu-validation.v1",state="passed" if outcome.wasSuccessful() else "failed",
        tests_run=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),skipped=len(outcome.skipped),
        elapsed_seconds=time.perf_counter()-started,source_file_sha256={p:sha(ROOT/p) for p in files},
        config_sha256=sha(doc/"S4_TFSS_FULL_LOCK.json"),
        numeric_component_smoke=dict(actual_Fourier_JND_TFSS_path=True,
            non_overlap_delta_difference=float((new[:,:,~overlap]-old[:,:,~overlap]).abs().max()),
            overlap_twice_reference_difference=float((new[:,:,overlap]-2*old[:,:,overlap]).abs().max()),
            Fourier_gradient_norm=float(pairs[1][2].norm()),float_linf=float(new.abs().max()),
            decoded_PNG_overlap_linf_bytes=int(np.abs(delta[target&other]).max()),
            decoded_PNG_outside_max_bytes=int(np.abs(delta[~target]).max())),
        environment=dict(torch=torch.__version__,numpy=np.__version__,device="CPU",torch_threads=1),
        official_metrics_disabled=True,artifact_ingest_disabled=True,
        GPU_smoke="NOT_RUN_pending_predecessor_acceptance_and_queue_admission",
        independent_scientific_review="NOT_CALLED_pending_current_commit_GPU_smoke",
        cache_capacity_fixture="Test-only mocked free-space return to exercise <1MiB total cache writes on local disk; not remote admission. All Q/R/model/native gradients and cache bytes are real.",
        evidence_limits=["Random-YAML CPU teacher is not formal pretrained teacher",
                         "CPU component tests do not validate full GPU Frozen/Adaptive outer training chain",
                         "No full dataset export or AP result generated","Retain all test artifacts; no cleanup"])
    (doc/"CPU_VALIDATION_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(log)
    print(json.dumps({k:report[k] for k in ("state","tests_run","failures","errors","skipped","elapsed_seconds","numeric_component_smoke")},ensure_ascii=False))
    if not outcome.wasSuccessful():raise SystemExit(1)

if __name__=="__main__":main()
