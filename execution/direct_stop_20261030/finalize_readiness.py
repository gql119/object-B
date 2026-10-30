"""Add actual retained remote log/status paths and run lean packet checker."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
packet=HERE/'SELECTED_PROTOCOL_SCIENTIFIC_REVIEW_PACKET.json'
v=json.loads(packet.read_text(encoding='utf8'));s=json.loads((HERE/'current_probes/s4_at_selected_native_20261030_gpu01/runspec.json').read_text())
v['pre_review_smoke']['retained_evidence_paths'] += [s['remote']['control_root']+'/console.log',s['remote']['output_root']+'/result/status.json',s['remote']['output_root']+'/result/smoke_summary.json']
packet.write_text(json.dumps(v,indent=2),encoding='utf8')
spec=importlib.util.spec_from_file_location('ready_checker',r'F:/autodl-tmp - 副本/ue_project/.agents/skills/pre-run-implementation-review/scripts/prerun_ready.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
r=module.validate_packet(v)
(HERE/'SELECTED_PROTOCOL_PACKET_READY.json').write_text(json.dumps(r,indent=2),encoding='utf8')
print(json.dumps(dict(ready=r['ready'],errors=r['errors'],packet_sha256=r['packet_sha256'])))
assert r['ready']
