import pytest
import torch

from scripts.cnlp_train_admission import average_eot_gradients


def sample(parent,cnlp):
    return dict(parent=torch.tensor(parent,dtype=torch.float64),
                cnlp=torch.tensor(cnlp,dtype=torch.float64),
                attack=torch.tensor(parent,dtype=torch.float64),
                ret=-torch.tensor(parent,dtype=torch.float64))


def test_eot_cancellation_does_not_report_mean_of_norms():
    result=average_eot_gradients([sample([1,0],[3,4]),sample([1,0],[-3,-4])])
    assert result['norm']['cnlp']==0
    assert result['cosine']['parent'] is None
    assert result['cosine_valid']['parent'] is False


def test_eot_mean_direction_before_norm_and_cosine():
    result=average_eot_gradients([sample([2,0],[6,8]),sample([0,2],[0,0])])
    assert result['norm']['cnlp']==5
    assert result['norm']['parent']==pytest.approx(2**.5)
    assert result['cosine']['attack']==pytest.approx(7/(5*2**.5))
    assert result['cosine']['ret']==pytest.approx(-7/(5*2**.5))


def test_nonfinite_eot_gradient_is_rejected():
    with pytest.raises(FloatingPointError):
        average_eot_gradients([sample([1,0],[float('nan'),0])])
