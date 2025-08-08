from .base import BaseLM

import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersLM(BaseLM):

    def __init__(self, model_hf_name, device_map="auto"):
        self.model = AutoModelForCausalLM.from_pretrained(model_hf_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_hf_name)

        self.vocab_token_size = self.tokenizer.vocab_size     # All tokens IDs including added tokens (w/o special tokens)
        self.vocab_size = len(self.tokenizer)           # All tokens IDs including added tokens (and special tokens)
        self.logits_size = self.model.config.vocab_size # Size of logits outputted by model's forward
        
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token = self.tokenizer.pad_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.pad_token_id


    def logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Any, ...]] = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[Any, ...]]]:
        

        # Should I allow the input to be a dict as in model() of transformers, w/ attention_mask and past_key_values optional ? I want to do this iff having only input_ids tensor as input still works
        real_len = 0

        self.model.eval()
        with torch.no_grad():
            # build kwargs dynamically
            model_inputs: Dict[str, Any] = {"input_ids": input_ids}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values

            # forward pass with caching enabled
            outputs = self.model(**model_inputs, use_cache=True)
            # shape: (*batch_size, seq_len, vocab_size)
            logits = outputs.logits

            # slice off only the last-token logits 
            next_token_logits = logits[..., real_len-1, :]    # shape: (*batch_size, vocab_size)

            # grab cache for next call (or None if model didn’t return it)
            new_past = getattr(outputs, "past_key_values", None)

        return next_token_logits, new_past  


    def sample(self, probs, temperature=1.0, top_k=None, top_p=None):
        """
        Given the probs of the next-token, sample the next-token.
        """
        
        # Make this whole function batch-size agnostic by flatting the logits in a shape (flat_B, V)


        if temperature<0:
            raise ValueError("temperature must be >= 0")
        if top_p is not None and (top_p > 1 or top_p < 0):
            raise ValueError("top-p must be between 0 and 1")
        if top_k is not None and top_k < 1:
            raise ValueError("top-k must be greater than 0")
        # add other raise for top_p, top_k, ...

        if temperature == 0:
            return probs.argmax(dim=-1, keepdim=True)
        
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True)


        if top_k is not None:
            probs = _apply_top_k(probs, top_k)
        if top_p is not None:
            probs = _apply_top_p(probs, top_p)
        # elif self.typical_p is not None:
        #     probs = _apply_typical_p(logits, self.typical_p)
        # if self.min_p is not None:
        #     probs = _apply_min_p(logits, self.min_p)

        return torch.multinomial(probs, num_samples=1) # shape: (batch_size, 1)

def _apply_top_k(probs, top_k): 
    # Find the top_k values and their indices
    topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
    # Create a mask that’s 1 for those top_k positions, 0 elsewhere
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk_idx, torch.ones_like(topk_vals))
    # Zero out everything else
    filtered = probs * mask
    
    row_sums = filtered.sum(dim=-1, keepdim=True)   # (B, 1)
    if torch.any(row_sums == 0):
        bad_rows = torch.nonzero(row_sums.squeeze(-1) == 0).squeeze(-1)
        raise RuntimeError(
            f"_apply_top_k: top-k + previous masks eliminated every token "
            f"in rows {bad_rows.tolist()} – the distribution sums to zero."
        )
    filtered = filtered / row_sums
    return filtered 

def _apply_top_p(probs, top_p):

    # Sort probabilities descending
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Mask all positions where cumulative sum > top_p (but keep at least one)
    exceed_mask = cumsum > top_p
    exceed_mask[..., 0] = False  # always keep the top‐1 token

    # Scatter mask back to original ordering
    full_mask = torch.zeros_like(probs, dtype=torch.bool)
    full_mask.scatter_(-1, sorted_idx, exceed_mask)

    # Zero out masked probs
    filtered = probs.masked_fill(full_mask, 0.0)
    # Renormalize
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered     