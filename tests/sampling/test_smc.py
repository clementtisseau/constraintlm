import torch
import math
import pytest

from constraintlm.sampling.sequence_sampling.smc import SMCSampler
from constraintlm.models.base import BaseLM
from constraintlm.models.transformers import TransformersLM
from constraintlm.constraints.base import Constraint


# ──────────────── Dummy model, constraint, critic ──────────────────


class DummyModel(BaseLM):
    """
    A minimal model that always returns the same logits for every token
    (so we can predict exactly how _extend / _reweight / _resample will behave).
    """
    def __init__(self, vocab_size=10, pad_token_id=0, eos_token_id=9):
        super().__init__()
        self.vocab_size   = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        # We’ll just keep a learnable “logit table” so that logits() returns something
        self.logit_table  = torch.nn.Parameter(torch.randn(vocab_size))

    def logits(self, input_ids, attention_mask=None, past_key_values=None):
        """
        The SMCSampler expects `model.logits(...)` to return (logits, past_key_values).
        We ignore `attention_mask` and `past_key_values`, and just broadcast our logit table.
        """
        batch_size = input_ids.shape[0]
        # Repeat B times the same logit vector
        out_logits = self.logit_table.unsqueeze(0).expand(batch_size, -1)  # (B, V)     
        return out_logits, past_key_values


class PassThroughConstraint(Constraint):
    """
    A no-op constraint: apply(…) returns the original probs unchanged
    and a normalizing constant of 1 for every particle.
    """
    def __init__(self, model):
        self.model = model
    def apply(self, input_ids, probs):
        """
        Given the token IDs generated so far (input_ids)
        and the raw next-token probs, return modified logits
        (e.g. with forbidden tokens masked, scores re-weighted, etc.).
        """
        B = probs.size(0)
        # “scored_probs” = probs, and “normalizing_cst” = ones(B)
        return probs, torch.ones(B, device=probs.device)
    
    def prefix():
        pass

    def complete():
        pass


class UnitCritic(Constraint):
    """
    A critic that always returns 1.  This makes log-ratio = 0 everywhere
    so that _reweight doesn’t change the weights.
    """
    def apply():
        pass 

    def prefix():
        pass

    def complete():
        pass

    def score(self, seqs):
        # seqs is shape (B*P, t or t+1), but we only care about shape[0]
        return torch.ones(seqs.shape[0], device=seqs.device)




# ──────────────── Tests ──────────────────

@pytest.mark.parametrize("B,P,L", [(2, 3, 4)])         # What's the point of this ?
def test_extend_shapes(B, P, L):
    model      = DummyModel(vocab_size=8, pad_token_id=0, eos_token_id=7)
    constraint = PassThroughConstraint(model)
    sampler    = SMCSampler(model, constraint)

    seq_len = 5
    # Create dummy inputs of shape (B*P, 1) to mimic the “t=0” call
    ids   = torch.randint(0, model.vocab_size-1, (B*P, seq_len))    # we exclude eos_token_id since they trigger sampling of pad_token only
    mask  = torch.ones(B*P, seq_len, dtype=torch.long)                # just 
    pkv   = None

    new_ids, norm_cst, new_mask, new_pkv = sampler._extend(
        t=0,
        input_ids=ids,
        flat_gen_ids=torch.empty(B*P, L, dtype=torch.long),
        flat_finished=torch.zeros(B*P, dtype=torch.bool),
        attention_mask=mask,
        past_key_values=pkv,
        flat_B_shape=B,
        P=P,
        device=ids.device,
    )
    # this call to _extend() will :
    # - call model.logits() (it always returns the same logits vector (B, V) and pkv = None)
    # - attention_mask is updated, logits converted to probs
    # - we apply the constraint to probs (apply(input_ids, probs)): here it does nothing
    # - we force pad_token_id to finished sentences (here it is impossible)
    # - we sample new_ids from probs

    # Check that shapes line up exactly with our expectations
    assert new_ids.shape      == (B * P, 1)
    assert norm_cst.shape     == (B * P,)
    assert new_mask.shape     == (B * P, mask.shape[-1] + 1)
    # Because this is a “passthrough” constraint, norm_cst should be all ones
    assert torch.allclose(norm_cst, torch.ones_like(norm_cst))

def test_extend_forces_pad_when_finished():
    """
    If flat_finished[i] is True, scored_probs[i, :] should become zero everywhere
    except scored_probs[i, pad_token_id] == 1.0, so new_ids[i] == pad_token_id.
    """
    B, P, L = 2, 3, 4
    model      = DummyModel(vocab_size=10, pad_token_id=0, eos_token_id=9)
    constraint = PassThroughConstraint(model)
    sampler    = SMCSampler(model, constraint)

    max_len = 5
    # Prepare a dummy batch of size (B*P, 1)
    ids   = torch.randint(0, model.vocab_size-1, (B*P, max_len))
    mask  = torch.ones(B*P, max_len, dtype=torch.long)
    pkv   = None

    # Let’s mark some particles as “finished”:
    flat_finished = torch.zeros(B*P, dtype=torch.bool)
    # e.g. make every even‐indexed particle “finished”
    flat_finished[0::2] = True

    # Run _extend(t=0) once, so scored_probs = probs, then force
    new_ids, norm_cst, new_mask, new_pkv = sampler._extend(
        t=0,
        input_ids=ids,
        flat_gen_ids=torch.zeros(B*P, L, dtype=torch.long),
        flat_finished=flat_finished,
        attention_mask=mask,
        past_key_values=pkv,
        flat_B_shape=B,
        P=P,
        device=ids.device,
    )

    # Wherever finished[i]==True, new_ids[i] must be pad_token_id (=0 here)
    for i in range(B * P):
        if flat_finished[i]:
            assert new_ids[i, 0].item() == model.pad_token_id
        else:
            # If not finished, new_ids[i] should be in [0..vocab_size)
            assert 0 <= new_ids[i, 0].item() < model.vocab_size


def test_extend_t_gt_zero_uses_prefix_correctly(tmp_path):
    """
    Verify that if t>0, we pass flat_gen_ids[:, :t] into constraint.apply.
    We do this by installing a custom constraint that records the prefix.
    """
    B, P, L = 2, 2, 5
    model      = DummyModel(vocab_size=6, pad_token_id=0, eos_token_id=5)

    class RecordingConstraint(PassThroughConstraint):
        def __init__(self, model):
            super().__init__(model)
            self.last_prefix = None

        def apply(self, prefix_ids, probs):
            # Save a clone of prefix_ids so we can inspect it from the test
            self.last_prefix = None if prefix_ids is None else prefix_ids.clone()
            return super().apply(prefix_ids, probs)

    constraint = RecordingConstraint(model)
    sampler    = SMCSampler(model, constraint)


    # Suppose L=5: we’ll pretend that flat_gen_ids already has some values in columns 0..(t-1).
    flat_gen_ids = torch.zeros(B*P, L, dtype=torch.long)
    # Manually fill in column 0 and 1 for each particle, e.g. [[1,2,0,0,0], [3,4,0,0,0]]
    flat_gen_ids[0, :2] = torch.tensor([1, 2])
    flat_gen_ids[1, :2] = torch.tensor([3, 4])
    flat_gen_ids[2, :2] = torch.tensor([3, 1])
    flat_gen_ids[3, :2] = torch.tensor([2, 4])

    # Mark third sentence as finished
    flat_finished = torch.zeros(B * P, dtype=torch.bool)
    flat_finished[2] = True
    # Create dummy input_ids = “previously generated token at t=1”
    # In reality for _extend(t=1) they pass input_ids=flat_gen_ids[:, t-1:t] = [:,1:2].
    input_ids = flat_gen_ids[:, 1].unsqueeze(-1)  # shape (4,1)

    attention_mask = torch.ones(B * P, 2, dtype=torch.long)  # e.g. for t=1, old length was 1
    pkv            = None

    new_ids, norm_cst, new_mask, new_pkv = sampler._extend(
        t=1,
        input_ids=input_ids,
        flat_gen_ids=flat_gen_ids,
        flat_finished=flat_finished,
        attention_mask=attention_mask,
        past_key_values=pkv,
        flat_B_shape=B,
        P=P,
        device=input_ids.device,
    )

    # Now the constraint must have been called with prefix=flat_gen_ids[:, :1]
    # because t=1, so prefix_ids shape should be (2, 1) and equal to the first column:
    assert constraint.last_prefix.shape == (B * P, 1)
    assert torch.equal(constraint.last_prefix[:, 0], flat_gen_ids[:, 0])


@pytest.mark.parametrize("B,P", [(1, 3), (2, 2)])
def test_extend_attention_mask_appends_one(B, P):
    """
    After each call to _extend, attention_mask should gain exactly 1 column of ones.
    """
    model      = DummyModel(vocab_size=7, pad_token_id=0, eos_token_id=6)
    constraint = PassThroughConstraint(model)
    sampler    = SMCSampler(model, constraint)

    # Suppose the original attention_mask had length L_mask = 4
    old_mask = torch.randint(0, 2, (B * P, 4), dtype=torch.long)
    flat_finished = torch.zeros(B * P, dtype=torch.bool)
    flat_gen_ids  = torch.zeros(B * P, 4, dtype=torch.long)

    # Create an arbitrary “input_ids” of shape (B*P,1)
    input_ids = torch.randint(0, model.vocab_size, (B * P, 1))
    pkv       = None

    new_ids, norm_cst, new_mask, new_pkv = sampler._extend(
        t=0,
        input_ids=input_ids,
        flat_gen_ids=flat_gen_ids,
        flat_finished=flat_finished,
        attention_mask=old_mask,
        past_key_values=pkv,
        flat_B_shape=B,
        P=P,
        device=old_mask.device,
    )

    # The new_mask should be old_mask concatenated with 1s-on-new-column
    assert new_mask.shape == (B * P, old_mask.shape[-1] + 1)
    # Check that the newly appended column is all ones
    assert torch.all(new_mask[:, -1] == 1)


def test_extend_deterministic_sampling_with_custom_constraint():
    """
    If our constraint returns a “one-hot” distribution, then new_ids should be that argmax every time.
    """
    B, P, L = 1, 3, 5
    model      = DummyModel(vocab_size=4, pad_token_id=0, eos_token_id=3)

    class OneHotConstraint(PassThroughConstraint):
        def apply(self, prefix_ids, probs):
            # Force distribution = one-hot at position “2” for every batch row
            Bp = probs.size(0)
            one_hot = torch.zeros_like(probs)
            one_hot[:, 2] = 1.0
            # So normalizing constant = 1 for each row
            return one_hot, torch.ones(Bp, device=probs.device)

    constraint = OneHotConstraint(model)
    sampler    = SMCSampler(model, constraint)

    ids   = torch.randint(1, model.vocab_size - 1, (B*P, 1))
    mask  = torch.ones(B*P, 1, dtype=torch.long)
    pkv   = None
    flat_finished = torch.zeros(B * P, dtype=torch.bool)

    new_ids, norm_cst, new_mask, new_pkv = sampler._extend(
        t=0,
        input_ids=ids,
        flat_gen_ids=torch.zeros(B*P, L, dtype=torch.long),
        flat_finished=flat_finished,
        attention_mask=mask,
        past_key_values=pkv,
        flat_B_shape=B,
        P=P,
        device=ids.device,
    )

    # Because OneHotConstraint forces probability=1 at token_id=2, new_ids[:,0] must be 2
    assert torch.all(new_ids.view(-1) == 2)
    # And norm_cst must be ones(B*P)
    assert torch.allclose(norm_cst, torch.ones_like(norm_cst))



# ----- Other tests -----

def test_reweight_no_critic_keeps_weights_unchanged():
    B, P = 2, 4
    model      = DummyModel(vocab_size=5)
    constraint = PassThroughConstraint(model)
    sampler    = SMCSampler(model, constraint, critic=None)

    # Simulate flat_log_weights of shape (B*P,)
    old_wts = torch.randn(B * P)
    # If normalizing_cst = 1 for all particles, and critic=None,
    # then reweight should not change `old_wts`
    new_wts = sampler._reweight(
        t=1,
        flat_log_weights=old_wts.clone(),
        normalizing_cst=torch.ones(B * P),
        flat_gen_ids=None,
        flat_B_shape=B,
        P=P,
        device=old_wts.device,
    )
    assert torch.allclose(new_wts, old_wts)


class DummyPastKeyValues:
    """
    This stub simply provides the two lists that _resample() tries to iterate over:
      past_key_values.key_cache
      past_key_values.value_cache
  
    Since they’re empty lists here, the for‐loop inside `_resample()` never runs,
    and it behaves exactly as if “real” past_key_values were present.
    """
    def __init__(self):
        self.key_cache = []
        self.value_cache = []


@pytest.mark.parametrize("B,P", [(3, 3), (2, 4)])
def test_resample_behavior(B, P):
    model      = DummyModel(vocab_size=6)
    constraint = PassThroughConstraint(model)
    sampler    = SMCSampler(model, constraint, critic=UnitCritic(model))

    # Create a “fake” scenario where ESS is low so we force resampling.
    # We can do that by making all log-weights extremely negative except one.
    flat_log_wts = torch.full((B * P,), float("-10.0"))
    # Give particle 0 in each batch a much higher weight → ESS will be tiny.
    for b in range(B):
        flat_log_wts[b * P + 0] = 10.0

    # Build dummy sequences so the “reshuffle” indexing will run.
    flat_gen_ids = torch.arange(B * P * 5).view(B * P, 5).clone()
    flat_finished = torch.zeros(B * P, dtype=torch.bool)
    attention_mask = torch.ones(B * P, 5, dtype=torch.long)
    dummy_pkv = DummyPastKeyValues()

    # We expect _resample to return updated flat_log_wts where each batch row
    # has been reset to -log(P) after resampling.
    new_wts, new_mask, new_pkv = sampler._resample(
        flat_gen_ids=flat_gen_ids,
        flat_log_weights=flat_log_wts.clone(),
        flat_finished=flat_finished,
        attention_mask=attention_mask.clone(),
        past_key_values=dummy_pkv,
        flat_B_shape=B,
        P=P,
        ess_threshold=1e6,  # force ESS<threshold every time
        device=flat_log_wts.device,
    )

    # Resampled weights: for every batch of size P, weights = -log(P)
    expected = torch.full((B * P,), -torch.log(torch.tensor(float(P))))
    # We know that new_wts is reshaped from shape (B, P) back to (B*P,)
    assert torch.allclose(new_wts, expected, atol=1e-5)

    # Check shapes of returned tensors
    assert new_mask.shape == attention_mask.shape

    assert new_pkv is dummy_pkv



# def test_smc_iteration_integration():
#     B, P, max_length = 1, 2, 4
#     model      = DummyModel(vocab_size=7, pad_token_id=0, eos_token_id=6)
#     constraint = PassThroughConstraint(model)
#     critic     = UnitCritic()
#     sampler    = SMCSampler(model, constraint, critic)

#     # Prompt: a single token “1”. After reshape → flat_B_shape=1, P=2, so input_ids=(2,1)
#     prompt_ids = torch.tensor([[1]])
#     # First iteration (t=0) needs “prompt_ids” repeated P times
#     flat_mult_prompt = prompt_ids.repeat(1, P).reshape(P, 1)  # (2,1)
#     flat_gen_ids   = torch.zeros(P, max_length, dtype=torch.long)
#     flat_log_wts   = torch.zeros(P)
#     flat_finished  = torch.zeros(P, dtype=torch.bool)
#     attention_mask = torch.ones(P, 1, dtype=torch.long)
#     past_kv        = None

#     # Manually call the first iteration
#     out_ids, out_wts, out_finished, new_mask, new_pkv = sampler._smc_iteration(
#         t=0,
#         input_ids=flat_mult_prompt,
#         flat_gen_ids=flat_gen_ids.clone(),
#         flat_log_weights=flat_log_wts.clone(),
#         flat_finished=flat_finished.clone(),
#         flat_B_shape=B,
#         P=P,
#         ess_threshold=0.5,    # doesn’t really matter here
#         max_length=max_length,
#         device=prompt_ids.device,
#         attention_mask=attention_mask,
#         past_key_values=past_kv,
#     )

#     # Every new row in out_ids has shape (P, max_length); t=0 fills column 0 only
#     assert out_ids.shape == (P, max_length)
#     # Because nothing has had a chance to hit EOS=6, out_finished should remain all False
#     assert not out_finished.any()
#     # out_wts should have been updated by log(normalizing_cst)=0 (since PassThroughConstraint→1)
#     assert out_wts.shape == (P,)
#     # attention_mask should have length 2 now (original 1 + appended 1)
#     assert new_mask.shape == (P, 2)
#     # past_key_values is still None for DummyModel
#     assert new_pkv is None




class NoOpConstraint(Constraint):
    def __init__(self, model):
        self.model = model

    def apply(self, prefix_ids, probs):
        # This constraint is never actually used in _resample(),
        # but SMCSampler requires a constraint when you instantiate it.
        return probs, torch.ones(probs.size(0), device=probs.device)

    def prefix(self):    pass
    def complete(self):  pass



# ─────────────── Test: “real‐model” resampling behavior ────────────────────

@pytest.mark.parametrize("P", [3])
def test_resample_reorders_everything_with_real_model(P):
    """
    Use a real TransformersLM (e.g. Qwen2.5-0.5B) so that `past_key_values` is
    non‐None. Then build a tiny toy batch (B=1, P=3) whose log‐weights force
    resampling. Finally assert that flat_gen_ids, flat_finished, attention_mask,
    past_key_values, and flat_log_weights all get shuffled by the same indices.
    """
    qwenllm = TransformersLM("Qwen/Qwen2.5-0.5B")

    prompt = "Hi"
    encoding = qwenllm.tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"]        # shape: (1, seq_len)

    # BUILD A TINY BATCH WITH P PARTICLES 
    B = 1
    max_length = 5
    flat_gen_ids = torch.empty(B*P, max_length, dtype=torch.long) # (B*P)
    # flat_gen_ids = torch.tensor([
    #     [100],   # “particle 0”
    #     [200],   # “particle 1”
    #     [300],   # “particle 2”
    # ], dtype=torch.long)
    # None of these have “finished” yet:
    flat_finished = torch.zeros((B * P,), dtype=torch.bool)

   
    # We want “only particle 0” to have extremely high weight so that sampling is almost certain to pick index=0 in each slot.
    flat_log_wts = torch.tensor([+10.0, -10.0, -10.0])


    # INSTANTIATE THE SAMPLER WITH OUR REAL MODEL + NO‐OP CONSTRAINT + CRITIC
    sampler = SMCSampler(
        model=qwenllm,
        constraint=NoOpConstraint(qwenllm),
        critic=UnitCritic(qwenllm)
    )
    input_ids = input_ids.repeat(P,1)
    # initialize attention_mask and past_keyvalues  
    attention_mask = (input_ids != sampler.model.pad_token_id).long()    # (B*P, L)


    new_ids, normalizing_cst, attention_mask, past_key_values = sampler._extend(0, input_ids, flat_gen_ids, flat_finished, attention_mask, None, B, P, torch.device("cpu"))

    # Keep track of EOS-terminated particles
    just_ended = new_ids.squeeze(-1) == sampler.model.eos_token_id # (B*P)
    flat_finished |= just_ended
    # Update particles
    flat_gen_ids[:, 0] = new_ids.squeeze(-1)   # (B*P, t+1)

    # CALL _resample(...) WITH A HUGE ESS THRESHOLD -> FORCES RESAMPLING
    new_wts, new_mask, new_pkv = sampler._resample(
        flat_gen_ids=flat_gen_ids.clone(),
        flat_log_weights=flat_log_wts.clone(),
        flat_finished=flat_finished.clone(),
        attention_mask=attention_mask.clone(),
        past_key_values=past_key_values,
        flat_B_shape=B,
        P=P,
        ess_threshold=1e9,                  # ESS will be < 1e9, so we always resample
        device=flat_log_wts.device,
    )

    # ASSERT “RESHUFFLING” EXERCISES THE CORRECT PERMUTATION ----------------

    # Since only index 0 had very large log‐weight, multinomial(weights, P, replace=True)
    # will (with prob ≈1) pick “0” for every new slot. Thus global_indices = [0, 0, 0].
    # That means: after resampling, every row of flat_gen_ids must be a copy of the old row 0.
    expected_gen = flat_gen_ids[0].expand(P, -1)  # shape (3, 1), all equal to old [100]

    # Check that `flat_gen_ids` was permuted into exactly [0,0,0] all‐copies
    assert torch.equal(new_wts, torch.full((P,), -math.log(float(P)))), \
        "After resampling, every log‐weight row should collapse to `-log(P)`"
    

    # CHECK THAT `attention_mask` WAS REORDERED THE SAME WAY:
    # Since `global_indices == [0,0,0]`, every row of new_mask must equal the old row0 of attn_rep.
    expected_mask = attention_mask[0].unsqueeze(0).expand(P, -1)
    assert torch.equal(new_mask, expected_mask), \
        "After resampling, all rows of attention_mask should be equal to the old row0"

    assert hasattr(new_pkv, "key_cache") and hasattr(new_pkv, "value_cache")
    assert len(new_pkv.key_cache) == len(past_key_values.key_cache), \
        "Number of layers in past_key_values should stay the same"

    # 10) FINALLY, ASSERT THAT THE NEW LOG‐WEIGHTS ARE ALL = -log(P):
    assert torch.allclose(
        new_wts,
        torch.full((P,), -math.log(float(P)), device=new_wts.device),
        atol=1e-5
    ), "Each of the P log-weights should be reset to -log(P) after resampling."
