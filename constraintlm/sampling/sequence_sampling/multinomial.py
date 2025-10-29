from .base import SequenceSampler

import torch

class MultinomialSeqSampler(SequenceSampler):

    def __init__(self, model, logits_processor=None):
        super().__init__(model)
        self.logits_processor = logits_processor
    
    def sample(self, prompt_ids, max_new_tokens, attention_mask=None, temperature=1.0, top_k=None, top_p=None, num_return_sequences=1):
        """
        Generates token sequences from a prompt using iterative sampling.

        Args:
            prompt_ids: Input token IDs to condition generation on. Shape (*batch_shape, seq_length).
            max_new_tokens: The maximum number of tokens to generate for each sequence.
            attention_mask: Optional mask to avoid attending to padding tokens. Shape (*batch_shape, seq_length).
            temperature: Softmax temperature for controlling randomness.
            top_k: Optional top-k cutoff for sampling.
            top_p: Optional nucleus (top-p) sampling cutoff.

        Returns:
            torch.Tensor: Generated token IDs with shape (*batch_shape, max_new_tokens).
        """

        # We need to reset logits_processor before sampling a new sentence
        self.logits_processor._seq_start_idx = None
        self.logits_processor._guide_states = {hash(tuple()): self.logits_processor.guide.initial_state}        # reset the _guide_states dictionary

        # self.model is your provider wrapper; the actual nn.Module is self.model.model
        nn_module = getattr(self.model, "model", self.model)  # fallback if you pass nn.Module directly
        was_training = nn_module.training
        nn_module.eval()

        try:
            with torch.inference_mode():
                pad_id = self.model.pad_token_id
                eos_id = self.model.eos_token_id

                *batch_shape, L = prompt_ids.shape
                device = prompt_ids.device

                flat_B = int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1

                # --- Flatten the batch_size and replicate prompts across num_return_sequences ---
                flat_prompt_ids = prompt_ids.reshape(flat_B, L)
                if num_return_sequences > 1:
                    flat_prompt_ids = flat_prompt_ids.unsqueeze(1).expand(flat_B, num_return_sequences, L)
                    flat_prompt_ids = flat_prompt_ids.reshape(flat_B * num_return_sequences, L)

                    if attention_mask is not None:
                        attention_mask = attention_mask.reshape(flat_B, L)
                        attention_mask = attention_mask.unsqueeze(1).expand(flat_B, num_return_sequences, L)
                        attention_mask = attention_mask.reshape(flat_B * num_return_sequences, L)
                else:
                    if attention_mask is not None:
                        attention_mask = attention_mask.reshape(flat_B, L)

                flat_BN = flat_prompt_ids.size(0)  # (flat_B * N)
                flat_gen_ids = torch.empty(flat_BN, max_new_tokens, dtype=torch.long, device=device)    # Future generated tokens
                flat_gen_ids = torch.full((flat_BN, max_new_tokens), pad_id, dtype=torch.long, device=device)
                flat_finished = torch.zeros(flat_BN, dtype=torch.bool, device=device)                   # Keep track of the EOS-terminated particles (B)                           
                if attention_mask is None:
                    attn_mask = (flat_prompt_ids != pad_id).long().to(device)
                else:
                    attn_mask = attention_mask.long().to(device)

                cur_input_ids = flat_prompt_ids
                cur_attn_mask = attn_mask
                past_key_values = None

                for t in range(max_new_tokens):
                    if torch.all(flat_finished):
                        break

                    if past_key_values is None:
                        next_token_logits, past_key_values = self.model.logits(cur_input_ids, cur_attn_mask) 
                    else:   # Feed just the last token sampled at the previous step
                        last_token = cur_input_ids[:, -1:].to(device)
                        next_token_logits, past_key_values = self.model.logits(last_token, cur_attn_mask, past_key_values) 
    
                    if self.logits_processor is not None:
                        next_token_logits = self.logits_processor.process_logits(cur_input_ids, next_token_logits)
                    
                    next_token_probs = torch.softmax(next_token_logits / max(temperature, 1e-8), dim=-1)
                    if flat_finished.any(): # only if at least one sentence is finished
                        next_token_probs[flat_finished, :] = 0.0
                        next_token_probs[flat_finished, pad_id] = 1.0
                    new_ids = self.model.sample(next_token_probs, temperature, top_k, top_p)

                    # Keep track of EOS-terminated particles
                    just_ended = new_ids.squeeze(-1) == eos_id  # (flat_B* P)   
                    flat_finished |= just_ended                                  
                    # Update particles 
                    flat_gen_ids[:, t] = new_ids.squeeze(-1)    # (flat_B, 1)

                    # Update input_ids and attention_mask
                    cur_input_ids = torch.cat([cur_input_ids, new_ids], dim=-1)
                    cur_attn_mask = torch.cat([cur_attn_mask, torch.ones((flat_BN, 1), device=device, dtype=torch.long)], dim=-1)

                if num_return_sequences == 1:
                    gen_ids = flat_gen_ids.view(*batch_shape, max_new_tokens)
                else:
                    gen_ids = flat_gen_ids.view(flat_B * num_return_sequences, max_new_tokens)
                return gen_ids
        finally:
            # Restore previous train/eval state. We store the training state in was_training, swith to inference mode, and make sure the training mode is applied back.
            nn_module.train(was_training)