from abc import ABC, abstractmethod
import torch

# ---------------   apply() should always consider the case where input_ids is torch.tensor([]) or None


class Constraint(ABC):
    def __init__(self, llm):
        #self.vocab = llm.vocab # ATTENTION: the size of logits, the size of vocab, the size of tokenizer, the size of tokenizer w/ special tokens might be different
        self.llm = llm
         

    @abstractmethod
    def apply(self,
              input_ids: torch.LongTensor,
              probs: torch.FloatTensor
             ) -> torch.FloatTensor:
        """
        Given the token IDs generated so far (input_ids)
        and the raw next-token probs, return modified probs
        (e.g. with forbidden tokens masked, scores re-weighted, etc.).
        """
        # apply() should always consider the case where input_ids is torch.tensor([]) or None
        pass

    @abstractmethod
    def prefix(self, 
               input_ids: torch.LongTensor,
               ) -> float:
        """
        Given the token IDs generated so far (input_ids), 
        return the score associated by the constraint.
        """
        pass

    @abstractmethod
    def complete(self, 
               input_ids: torch.LongTensor,
               ) -> float:
        """
        Given the token IDs of a complete (EOS-terminated) sequence (input_ids), 
        return the score associated by the constraint.
        """
        pass

    def score(self, 
               input_ids: torch.LongTensor,
               ) -> float:
        pass