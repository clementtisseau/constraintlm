from .base import SequenceSampler

import torch
import math
from dataclasses import dataclass
# from ...constraints.base import Constraint

from ...processors.structured import CLMLogitsProcessor

@dataclass
class State:
    new_ids: torch.Tensor
    flat_gen_ids: torch.Tensor
    flat_log_weights: torch.Tensor
    flat_finished: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: torch.Tensor


class SMCSampler(SequenceSampler):

    def __init__(self, model, logits_processor: CLMLogitsProcessor, critic = None):      # the logits_processor is not optional. critic is optional though.
        super().__init__(model)
        self.logits_processor = logits_processor
        self.critic = critic


    def sample(self, prompt_ids, max_length, num_particles, ess_threshold):
        """
        prompt_ids: Tensor of shape (..., seq_length)
        Returns:
        out_ids:  Tensor of shape (*batch_shape, num_particles, max_length)
        out_wts:  Tensor of shape (*batch_shape, num_particles)
        """
        if ess_threshold <= 1: 
            print("ESS always is >= 1, particles won't be resampled.")
        L     = prompt_ids.size(-1) #seq_length
        B_shape = list(prompt_ids.shape[:-1])
        P = num_particles
        device = prompt_ids.device

        # Flatten prompt_ids for a batch-size-agnostic function
        flat_B_shape = int(torch.prod(torch.tensor(B_shape))) if B_shape else 1
        flat_prompt_ids = prompt_ids.reshape(flat_B_shape, L)   # (B, L)
        # Duplicate flat_prompt_ids for each particle
        flat_mult_prompt_ids = (
            flat_prompt_ids.unsqueeze(1)        # (B, 1, L)
                    .repeat(1, P, 1)            # (B, P, L)
                    .reshape(flat_B_shape*P, L) # (B*P, L)
            )
        
        # Generated particles
        flat_gen_ids = torch.empty(flat_B_shape*P, max_length, dtype=torch.long, device=device) # (B*P)

        # Keep track of the EOS-terminated particles
        flat_finished = torch.zeros(flat_B_shape*P,    dtype=torch.bool, device=device)         # (B*P)
        # log-weights of all particles
        flat_log_weights  = torch.zeros(flat_B_shape*P, dtype=torch.float, device=device)   # (B*P)
        

        # t = 0 : initialize attention_mask and past_keyvalues  
        attention_mask = (flat_mult_prompt_ids != self.model.pad_token_id).long()    # (B*P, L)     
        # maybe that I should delete this, since in transformers tokenizer(sentences, padding=True) return input_ids, attention_mask
        # however keeping it isn't a problem

        flat_gen_ids, flat_log_weights, flat_finished, attention_mask, past_key_values = self._smc_iteration(0, 
                                           flat_mult_prompt_ids, 
                                           flat_gen_ids, 
                                           flat_log_weights, 
                                           flat_finished, 
                                           flat_B_shape, 
                                           P, 
                                           ess_threshold, 
                                           max_length,
                                           device, 
                                           attention_mask=attention_mask, 
                                           past_key_values=None)
        
        
        for t in range(1, max_length):
            print(t)
            flat_gen_ids, flat_log_weights, flat_finished, attention_mask, past_key_values = self._smc_iteration(t, 
                                           flat_gen_ids[:, t-1].unsqueeze(-1), 
                                           flat_gen_ids, 
                                           flat_log_weights, 
                                           flat_finished, 
                                           flat_B_shape,    # fixed
                                           P,               # fixed
                                           ess_threshold,   # fixed
                                           max_length,      # fixed
                                           device,          # fixed
                                           attention_mask=attention_mask, 
                                           past_key_values=past_key_values)

        gen_ids = flat_gen_ids.reshape(*B_shape, P, max_length) 
        final_log_weights = flat_log_weights.reshape(*B_shape, P)

        return gen_ids, final_log_weights    # (*batch_shape, P, L), (*batch_shape, P)
    
    def _smc_iteration(self, t, input_ids, flat_gen_ids, flat_log_weights, flat_finished, flat_B_shape, P, ess_threshold, max_length, device, attention_mask=None, past_key_values=None):
        
        # 1. ----- Extend -----
        new_ids, normalizing_cst, attention_mask, past_key_values = self._extend(t, input_ids, flat_gen_ids, flat_finished, attention_mask, past_key_values, flat_B_shape, P, device)

        # Keep track of EOS-terminated particles
        just_ended = new_ids.squeeze(-1) == self.model.eos_token_id # (B*P)
        flat_finished |= just_ended
        # Update particles
        flat_gen_ids[:, t] = new_ids.squeeze(-1)   # (B*P, t+1)

        # 2. ----- Reweight -----
        flat_log_weights = self._reweight(t, flat_log_weights, normalizing_cst, flat_gen_ids, flat_B_shape, P, device)

        # 3. ----- Resample -----
        flat_gen_ids, flat_log_weights, flat_finished, attention_mask, past_key_values = self._resample(flat_gen_ids, flat_log_weights, flat_finished, attention_mask, past_key_values, flat_B_shape, P, ess_threshold, device)

        return flat_gen_ids, flat_log_weights, flat_finished, attention_mask, past_key_values
    




    def _extend(self, t, input_ids, flat_gen_ids, flat_finished, attention_mask, past_key_values, flat_B_shape, P, device):
        logits, past_key_values = self.model.logits(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            past_key_values = past_key_values
            )
        attention_mask = torch.cat([attention_mask, torch.ones((flat_B_shape*P, 1), dtype=attention_mask.dtype, device=device)], dim=-1)
        
        if t==0:
            scored_logits = self.logits_processor.process_logits(torch.empty((flat_B_shape*P, 0), dtype=torch.long), logits)
        else: 
            scored_logits = self.logits_processor.process_logits(flat_gen_ids[:,:t], logits)
        
        normalizing_cst = torch.exp(scored_logits).sum(-1) / torch.exp(logits).sum(-1)
        scored_probs = torch.softmax(scored_logits, dim=-1)

        #assert torch.all(scored_probs.sum(-1) == 1), "Probabilities doesn't sum to 1."
        assert not torch.any(scored_probs.sum(-1) == 0), "Probabilities sum to 0."
        assert torch.all(scored_probs.sum(-1) > 0), "Probabilities <= 0."
        scored_probs[flat_finished, :] = 0.0
        scored_probs[flat_finished, self.model.pad_token_id] = 1.0
        #assert torch.all(scored_probs.sum(-1) == 1), "Probabilities doesn't sum to 1."
        assert not torch.any(scored_probs.sum(-1) == 0), "Probabilities sum to 0."
        assert torch.all(scored_probs.sum(-1) > 0), "Probabilities <= 0."


        # Here we have a problem, what happens if we sample a token_id that is greater than the vocab_token_size (<= logits_size)
        new_ids = torch.multinomial(scored_probs, num_samples=1)           # (B*P, 1)
        # Does it make sense to change the scored_probs where a sentence is over, and sample from those changed scored_probs with a forcing value of pad_token_id ?
        # Couldn't we, instead, sample from scored_probs (even for the sentences that are finished), and then modify new_ids such that in the finished sentence we modify the sampled token_id to be pad_token_id
        # Both techniques are doing the same thing, which one is more efficient?
        return new_ids, normalizing_cst, attention_mask, past_key_values
    

    def _reweight(self, t, flat_log_weights, normalizing_cst, flat_gen_ids, flat_B_shape, P, device):
        #print(f"{t} before reweight: {torch.exp(flat_log_weights)}")
        L_eff = normalizing_cst # (B*P)
        log_L_eff = torch.log(L_eff.clamp_min(1e-30))

        if self.critic == None:
            log_ratio = torch.zeros(flat_B_shape*P, device=device) # (B*P)
        elif t == 0:
            ratio = self.critic.score(flat_gen_ids[:,:t+1])
            log_ratio = torch.log(ratio)
        else:
            ratio = self.critic.score(flat_gen_ids[:,:t+1]) / self.critic.score(flat_gen_ids[:,:t]) # (B*P)
            log_ratio = torch.log(ratio) 

        flat_log_weights += log_L_eff + log_ratio                   # (B*P)
        #print(f"{t} after reweight: {torch.exp(flat_log_weights)}")
        return flat_log_weights

    def _resample(self, flat_gen_ids, flat_log_weights, flat_finished, attention_mask, past_key_values, flat_B_shape, P, ess_threshold, device):
        log_weights = flat_log_weights.reshape(flat_B_shape, P) # (B, P) 
        lse1 = torch.logsumexp(log_weights,     dim=-1)         # (B)
        lse2 = torch.logsumexp(2 * log_weights, dim=-1)         # (B)
        ess  = torch.exp(2 * lse1 - lse2)                       # (B)
        #print(f"ESS: {ess}")
        resampling_mask = ess < ess_threshold                   # (B)
        
        if torch.any(resampling_mask):
            print("Resampling...")
            weights = torch.exp(log_weights - log_weights.max(dim=-1, keepdim=True).values)     # (B, P)

            resampling_indices = torch.multinomial(weights, num_samples=P, replacement=True)    # (B, P)
            resampling_indices[~resampling_mask, :] = torch.arange(P, device=device)            # where we don't need to resample, we keep the same order
            batch_idx = torch.arange(flat_B_shape, device=device).unsqueeze(1)                  # (B, 1)

            global_indices = (batch_idx * P + resampling_indices)   # (B, P)        [[0 + 0, 0 + 1, ..., 0 + (P-1)], [P + 0, P + 1, ..., P + (P-1)], ..., [(B-1)P + 0, (B-1)P + 1, ..., (B-1)P + (P-1)]]
            global_indices = global_indices.reshape(-1)             # (B*P,)

            # Reorder the tensors according to the resampling (use global indices everywhere might be much more efficient)
            # flat_gen_ids = flat_gen_ids.view(flat_B_shape, P, max_length)[batch_idx, resampling_indices].reshape(flat_B_shape*P, max_length)     # (B*P, max_length)
            # flat_finished = flat_finished.view(flat_B_shape, P)[batch_idx, resampling_indices].reshape(flat_B_shape*P)
            flat_gen_ids = flat_gen_ids[global_indices]     # (B*P, max_len)
            flat_finished = flat_finished[global_indices]   # (B*P)
            attention_mask = attention_mask[global_indices]
            # (use global indices everywhere might be much more efficient)
            #attention_mask = attention_mask.view(flat_B_shape, P, -1)[batch_idx, resampling_indices].reshape(flat_B_shape*P, -1)
            
            # Is this one working?
            # past_key_values.key_cache[0].shape        # (B*P, n_heads, seq_len, head_dim)
            for layer_idx in range(len(past_key_values.key_cache)):
                kc = past_key_values.key_cache[layer_idx].unflatten(0, (flat_B_shape, P))     # (B, P, n_heads, seq_len, head_dim)
                vc = past_key_values.value_cache[layer_idx].unflatten(0, (flat_B_shape, P))   # (B, P, n_heads, seq_len, head_dim)
                kc = kc[batch_idx, resampling_indices].contiguous()
                vc = vc[batch_idx, resampling_indices].contiguous()
                past_key_values.key_cache[layer_idx]  = kc.view(flat_B_shape * P, *kc.shape[2:])
                past_key_values.value_cache[layer_idx] = vc.view(flat_B_shape * P, *vc.shape[2:])
            
            #past_key_values.reorder_cache(global_indices)


            mean_weights = log_weights.exp().mean(dim=1)           # (B,)
            log_mean_weights   = torch.log(mean_weights)           # (B,)
            # use log_mean_weights = (torch.logsumexp(log_weights, dim=1) - math.log(P)) instead ?? More stable but are we changing the behavior of the algo ?
            mask2d = resampling_mask.unsqueeze(1).expand(-1, P)    # (B, P) rows are 1 only or 0 only
            log_weights = torch.where(mask2d, log_mean_weights.unsqueeze(1), log_weights)  # (B, P)
            flat_log_weights = log_weights.view(-1)

            # # build a per‐sample mask by repeating the batch‐mask P times:
            # mask_rep = resampling_mask.unsqueeze(1).expand(-1, P).reshape(-1)       #(B*P)
            # flat_log_weights = flat_log_weights.masked_fill(mask_rep, -math.log(float(P)))

        return flat_gen_ids, flat_log_weights, flat_finished, attention_mask, past_key_values