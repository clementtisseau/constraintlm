from .base import Constraint
import torch

# This class can be made more efficient using FSMConstraint.from_regex() 
# ^(?:\S{1,n}(?:\s+\S{1,n})*)?$                     or ^\s*(?:\S{1,n}(?:\s+\S{1,n})*)?\s*$ to allow leading and trailing whiteSpace chars.
# (?: … ) : groups without capturing
# \S{1,n} : between 1 and n not whiteSpace characters
# \s+ : 1 or more whiteSpace characters [ \t\n\r\f\v]
# \s+\S{1,n} : 1 or more whiteSpace char, followed by 1, ..., n not whiteSpace char.

# §§§§§§§§§§§§§§§§§§   apply() and score() (thus prefix(), complete() too) should always consider the case where input_ids is torch.tensor([]) or None

# The way this class is defined, '1000' will be considered as a word of length 4, while '1.000' of length 5, 'word' as length 4 and 'words?' as length 5.

class LengthWord(Constraint):
    def __init__(self, model, N):
        self.model = model
        self.N = N

    def apply(self, input_ids, probs):
        batch_shape = probs.shape[:-1]
        probs_size = probs.shape[-1]          # size outputted by the model 
        vocab_size = self.model.vocab_size      # which is different from vocab_size
        device = probs.device

        if input_ids is None or input_ids.numel() == 0:
            input_ids = torch.empty(*batch_shape, 0, dtype=torch.long, device=device)
        
        seq_len = input_ids.shape[-1]
        batch_size = int(torch.tensor(input_ids.shape[:-1]).prod())  # safe even if shape is empty (when input_ids is None)

        # flatten batch dims so we can loop
        flat_ids    = input_ids.view(batch_size, seq_len)       # (B, L)
        flat_probs = probs.view(batch_size, probs_size)      # (B, V)
        flat_mask   = torch.zeros_like(flat_probs)             # (B, V)

        for i, seq in enumerate(flat_ids):
            # for each possible next token compute penalty
            penalties = torch.full((probs_size,), 0.0, device=device)   # We need to do this to deal w/ the fact that while the model has a vocab of vocab.size, it outputs in a higher dim
            for tok in range(vocab_size):
                # build candidate sequence of shape (1, L+1)
                new_seq = torch.cat([
                    seq, 
                    torch.tensor([tok], device=device)
                ], dim=0).unsqueeze(0)

                # score returns tensor of shape (1,) with 0 or -inf
                pen = self.score(new_seq).item()
                penalties[tok] = pen

            flat_mask[i] = penalties

        # reshape mask back to (*batch_shape, logits_size)
        mask = flat_mask.view(*batch_shape, probs_size)

        unnormalized_probs = probs * mask               #(B, V)
        normalizing_cst = unnormalized_probs.sum(-1)    #(B)
        return unnormalized_probs / (normalizing_cst.unsqueeze(1) + 1e-8), normalizing_cst
        
    def prefix(self, input_ids):
        if input_ids is None or input_ids.numel() == 0:
            return 1
        # Flatten batch dimensions
        batch_shape = input_ids.shape[:-1]
        seq_length = input_ids.shape[-1]
        flat_ids = input_ids.view(-1, seq_length)
        
        # Decode each sequence to text
        #Big issue here, a special token such as <|endoftext|> will count as a long word
        texts = self.model.tokenizer.batch_decode(flat_ids, skip_special_tokens=True)
        
        # Determine penalty per sequence
        penalties = []
        for text in texts:
            # Split on whitespace to get words
            words = text.split()
            # Check if any word is too long
            if any(len(word) > self.N for word in words):
                penalties.append(0.0)
            else:
                penalties.append(1.0)
        
        # Convert to tensor and reshape to original batch shape
        penalties = torch.tensor(penalties, dtype=torch.float, device=input_ids.device)
        return penalties.view(batch_shape)
    
    def complete(self, input_ids):
        if input_ids is None or input_ids.numel() == 0:
            return 1
        # Flatten batch dimensions
        batch_shape = input_ids.shape[:-1]
        seq_length = input_ids.shape[-1]
        flat_ids = input_ids.view(-1, seq_length)
        
        # Decode each sequence to text
        #Big issue here, a EOS_token such as <|endoftext|> will count as a long word
        texts = self.model.tokenizer.batch_decode(flat_ids, skip_special_tokens=True)
        
        # Determine penalty per sequence
        penalties = []
        for text in texts:
            # Split on whitespace to get words
            words = text.split()
            # Check if any word is too long
            if any(len(word) > self.N for word in words):
                penalties.append(0.0)
            else:
                penalties.append(1.0)
        
        # Convert to tensor and reshape to original batch shape
        penalties = torch.tensor(penalties, dtype=torch.float, device=input_ids.device)
        return penalties.view(batch_shape)
    
    def score(self, input_ids):
        if input_ids is None or input_ids.numel() == 0:
            return 1
        # Flatten batch dimensions
        batch_shape = input_ids.shape[:-1]
        seq_length = input_ids.shape[-1]
        flat_ids = input_ids.view(-1, seq_length)
        
        # Decode each sequence to text
        #Big issue here, a EOS_token such as <|endoftext|> will count as a long word
        texts = self.model.tokenizer.batch_decode(flat_ids, skip_special_tokens=True)
        
        # Determine penalty per sequence
        penalties = []
        for text in texts:
            # Split on whitespace to get words
            words = text.split()
            # Check if any word is too long
            if any(len(word) > self.N for word in words):
                penalties.append(0.0)
            else:
                penalties.append(1.0)
        
        # Convert to tensor and reshape to original batch shape
        penalties = torch.tensor(penalties, dtype=torch.float, device=input_ids.device)
        return penalties.view(batch_shape)

