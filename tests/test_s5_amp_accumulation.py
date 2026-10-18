from scripts.s5_victim_smoke import observe_forced_overflow
import pytest

def state():
    return dict(forced_batch=2,forced_after_actual_updates=1,recovered=False)

def row(batch,attempt=False,skip=False,backoff=False,unchanged=None):
    return dict(total_batches=batch,optimizer_attempt_observed=attempt,
                native_scaler_nonfinite_skip=skip,backoff=backoff,
                parameters_unchanged_on_skip=unchanged)

def test_delayed_native_attempt_and_recovery():
    s=state();rows=[row(1,True),row(2)]
    observe_forced_overflow(s,rows,1)
    assert not s['recovered'] and 'forced_optimizer_attempt_batch' not in s
    rows.append(row(3,True,True,True,True))
    observe_forced_overflow(s,rows,1)
    assert s['forced_optimizer_attempt_batch']==3 and not s['recovered']
    rows.append(row(4,True))
    observe_forced_overflow(s,rows,2)
    assert s['recovered']

def test_same_batch_native_skip_is_valid():
    s=state();observe_forced_overflow(s,[row(1),row(2,True,True,True,True)],1)
    assert s['forced_optimizer_attempt_batch']==2 and not s['recovered']

@pytest.mark.parametrize('skip,backoff,unchanged',[(False,False,False),(True,False,True),(True,True,False)])
def test_real_failure_is_still_rejected(skip,backoff,unchanged):
    with pytest.raises(AssertionError):
        observe_forced_overflow(state(),[row(1),row(2,True,skip,backoff,unchanged)],1)
