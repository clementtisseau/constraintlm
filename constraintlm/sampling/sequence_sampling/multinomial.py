from .base import SequenceSampler

import torch

class MultinomialSeqSampler(SequenceSampler):

    def __init__(self, model, constraint=None):
        super().__init__(model, constraint)
    
    def sample(self, prompt_ids, max_length, temperature=1.0, top_k=None, top_p=None):

        L     = prompt_ids.size(-1)             # seq_length
        B_shape = list(prompt_ids.shape[:-1])   # (*batch_shape)
        device = prompt_ids.device
        # Flatten prompt_ids for a batch-size-agnostic function
        flat_B_shape = int(torch.prod(torch.tensor(B_shape))) if B_shape else 1
        flat_prompt_ids = prompt_ids.reshape(flat_B_shape, L)       # (B, L)
        # Generated tokens
        flat_gen_ids = torch.empty(flat_B_shape, max_length, dtype=torch.long, device=device)
        # Keep track of the EOS-terminated particles
        flat_finished = torch.zeros(flat_B_shape,    dtype=torch.bool, device=device)       # (B)
        
        # --- t=0 ---
        attn_mask = (flat_prompt_ids != self.model.pad_token_id).long()      # (B, L) 
        next_token_logits, past_key_values = self.model.logits(flat_prompt_ids, attn_mask)
        attn_mask = torch.cat([attn_mask, torch.ones((flat_B_shape, 1))], dim = -1)
        next_token_probs = torch.softmax(next_token_logits, dim=-1) # model.sample() samples from probs
        if self.constraint is not None:
            next_token_probs, _ = self.constraint.apply(None, next_token_probs)
        new_ids = self.model.sample(next_token_probs, temperature, top_k, top_p)

        # Keep track of EOS-terminated particles
        just_ended = new_ids.squeeze(-1) == self.model.eos_token_id # (B)
        flat_finished |= just_ended
        # Update particles 
        flat_gen_ids[:, 0] = new_ids.squeeze(-1)   # (B, 1)

        for t in range(1, max_length):
            print(t)
            next_token_logits, past_key_values = self.model.logits(new_ids, attn_mask, past_key_values)
            attn_mask = torch.cat([attn_mask, torch.ones((flat_B_shape, 1))], dim = -1)
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            if self.constraint is not None:
                next_token_probs, _ = self.constraint.apply(flat_gen_ids[:,:t], next_token_probs)

            next_token_probs[flat_finished, :] = 0.0      
            next_token_probs[flat_finished, self.model.pad_token_id] = 1.0
            new_ids = self.model.sample(next_token_probs, temperature, top_k, top_p)

            # Keep track of EOS-terminated particles
            just_ended = new_ids.squeeze(-1) == self.model.eos_token_id # (flat_B* P)   # self.model.eos_token_id is not defined
            flat_finished |= just_ended                                  
            # Update particles 
            flat_gen_ids[:, t] = new_ids.squeeze(-1)   # (flat_B, 1)

        gen_ids = flat_gen_ids.reshape(*B_shape, max_length)
        return gen_ids
