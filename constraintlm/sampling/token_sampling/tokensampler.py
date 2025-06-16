import torch

class TokenSampler:

    def __init__(
                self,
                #constraint = None,
                temperature = 1.0,
                top_k = None,
                top_p = None,
                #typical_p = None,
                #min_p = None
            ):
        
        

        if temperature<0:
            raise ValueError("temperature must be >= 0")
        if top_p is not None and (top_p > 1 or top_p < 0):
            raise ValueError("top-p must be between 0 and 1")
        # add other raise for top_p, top_k, ...

        #self.constraint = constraint
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # self.typical_p = typical_p
        # self.min_p = min_p

    def sample(self, logits):
        """
        Given the probs of the next-token, sample the next-token.
        """
        
        # Make this whole function batch-size agnostic by flatting the logits in a shape (flat_B, V)

        if self.temperature == 0:
            return logits.argmax(dim=-1)
        logits = logits / self.temperature      # shape: (batch_size, vocab_size)


        if self.top_k is not None:
            probs = _apply_top_k(logits, self.top_k)
        if self.top_p is not None:
            probs = _apply_top_p(logits, self.top_p)
        # elif self.typical_p is not None:
        #     probs = _apply_typical_p(logits, self.typical_p)
        # if self.min_p is not None:
        #     probs = _apply_min_p(logits, self.min_p)
        
        if self.top_k is None and self.top_p is None:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float) 

        return torch.multinomial(probs, num_samples=1) # shape: (batch_size, 1)

def _apply_top_k(logits, top_k): 
    new_logits = torch.full_like(logits, float('-inf'))
    topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
    new_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)

    return torch.softmax(new_logits, dim=-1)  

def _apply_top_p(logits, top_p):

    # Create a mask where False: we'll keep the logit, True: we'll set it to -inf
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    sorted_probs, indices_sorted_probs = torch.sort(probs, dim=-1, descending=True) 
    cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_boolean_mask = (cumsum_sorted_probs > top_p)
    sorted_boolean_mask[..., 0] = False    

    boolean_mask = torch.zeros_like(logits, dtype=torch.bool)
    boolean_mask.scatter_(dim=-1, index=indices_sorted_probs, src=sorted_boolean_mask)

    new_logits = torch.clone(logits)
    new_logits[boolean_mask] = float('-inf')

    return torch.softmax(new_logits, dim=-1)


if __name__ == "__main__":
    # stuff under this block only runs when you do:
    #   python script.py
    # but NOT when you do:
    #   import script
    # debugging test : ipython, then run -d script.py, or python -i script.py, or python -m pdb script.py
    logits = torch.tensor([[1., 0.3, 0.01], [1., 1., 1.]])
    probs = _apply_top_p(logits, 0.8)
    print(probs)