import torch
from constraintlm.sampling.token_sampling.tokensampler import _apply_top_k, _apply_top_p


def test_apply_top_k_basic():
    probs = _apply_top_k(torch.tensor([1., 4., 2., 3.]), 2)

    assert probs[0] == 0 and probs[2] == 0
    assert probs[1] + probs[3] == 1     

def test_apply_top_p_basic():
    probs = _apply_top_p(torch.tensor([[0.1,2.0,0.5,1.1]]), top_p=0.5)
    # only the highest-prob index 1 remains
    assert probs[0].argmax() == 1
    assert torch.isclose(probs.sum(-1), torch.tensor([1.0]))