"""At most three extra diagnostic samples; no speed winner from single runs."""
import time
from finite_implementation_suite import deadline,event,cli,finish


if __name__=='__main__':
    cases=[('b48single','bench',48),('b64single','bench',64),('profile36','profile',36)]
    try:
        for label,kind,batch in cases:
            if time.time()+1020>deadline:
                event(state='OPTIONAL_SAMPLE_SKIPPED_FOR_CAMPAIGN_MARGIN',label=label);break
            cli('launch','--label',label,'--prepared',f's4_at_speed_{kind}_coriginal_clean4_b{batch}_r1_20261030_gpu01.json',
                '--tag',f's4_at_{kind}_b{batch}_original_20261030_direct01')
            finish(label)
        event(state='FINITE_REMAINING_SAMPLES_FINISHED',single_samples_do_not_establish_speedup=True)
    except BaseException as exc:
        event(state='FINITE_OPTIONAL_SUITE_STOPPED_NO_FURTHER_DISPATCH',error=repr(exc));raise
