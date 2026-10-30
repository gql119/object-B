"""Continue only unlaunched remaining probes after prelaunch Git TLS failure."""
import json
from finish_fallback_preflight import HERE,cli,finish

if __name__=='__main__':
    assert (HERE/'current_probes/s4_at_selected_native_20261030_gpu01/pull.json').is_file()
    path=HERE/'current_probes/s4_at_selected_baseline_20261030_gpu01'
    assert not path.exists(),'Baseline already prepared; inspect instead of duplicate launch'
    record=dict(state='PREPARATION_CONNECTION_FAILURE_NO_RUNSPEC_OR_LAUNCH',
        error='git ls-remote TLS unexpected eof while reading',source_unchanged=True,
        normal_connection_retry_only=True,TLS_settings_changed=False,scientific_failure=False)
    (HERE/'BASELINE_PREPARATION_TLS_FAILURE.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    for kind,tag in [('baseline','s4_at_selected_baseline_20261030_gpu01'),('resume','s4_at_selected_phase_20261030_gpu01')]:
        cli('launch','--tag',tag,'--kind',kind);finish(tag)
    print('CURRENT_SOURCE_REMAINING_GPU_GATES_FINISHED_NO_FORMAL_LAUNCH',flush=True)
