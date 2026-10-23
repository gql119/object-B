"""First evaluation only of an already completed Full victim; never trains."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_s4_tfss_full import evaluate_completed,resource_admission
from scripts.s4_full_report import finalize
from ue_framework.s4_full_contract import load_full_config,verify_queue,sha

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    p.add_argument("--lock",required=True);p.add_argument("--admission",required=True)
    a=p.parse_args();out=Path(a.output).resolve();lock=json.loads(Path(a.lock).read_text())
    if not out.is_relative_to(Path(lock["remote_root"])/"outputs"):
        raise ValueError("Independent output boundary required")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip():
        raise ValueError("Dirty source")
    state=json.loads((out/"pipeline_status.json").read_text())
    review=json.loads((out/"review_snapshot.json").read_text())
    if (state["source_commit"]!=commit or sha(a.lock)!=sha(out/"config_lock_snapshot.json")
        or review.get("source_commit")!=commit or review.get("pre_run_result")!="pass"):
        raise ValueError("Training/evaluation source and review identity mismatch")
    if json.loads((out/"dataset_integrity.json").read_text()).get("state")!="passed":
        raise ValueError("Dataset audit missing")
    receipt=json.loads(Path(a.admission).read_text())
    verify_queue(lock,receipt,source_commit=commit,config_sha256=sha(a.lock))
    resource_admission()
    verify_queue(lock,receipt,source_commit=commit,config_sha256=sha(a.lock))
    cfg=load_full_config(ROOT,lock)
    evaluate_completed(cfg,out,lock)
    finalize(out,ROOT,commit,lock)

if __name__=="__main__":main()
