import torch
import math
from typing import List

from outlines.processors import OutlinesLogitsProcessor, GuideLogitsProcessor
from .guide import CLMCFGGuide


class CLMLogitsProcessor(OutlinesLogitsProcessor):
    def __init__(self, llm):
        self.llm = llm


class RPNLogitsProcessor(CLMLogitsProcessor):

    def __init__(self, llm):
        super().__init(llm)

    def process_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.Tensor:
        pass




class CLMCFGLogitsProcessor(GuideLogitsProcessor):        # this is not parallelized at all, could we improve this?  

    def __init__(self, cfg_str: str, tokenizer, llm, tensor_library_name):
        self.llm = llm
        cfg_guide = CLMCFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(
            tokenizer=tokenizer,
            guide=cfg_guide,
            tensor_library_name=tensor_library_name,
        )

    def process_logits(self, input_ids, logits):
        """
        Parameters
        ----------
        input_ids
            The ids of the tokens of the existing sequences.
        logits
            The logits for the current generation step.

        Returns
        -------
        TensorType
            The biased logits.

        """
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0]) 

        sequence_states: List = []  # vector of states corresponding to `input_ids`

        for seq_ids in input_ids: 
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(self.tensor_adapter.to_list(gen_ids)))

            if curr_state_key not in self._guide_states: # pragma: no cover
                prev_state = self._guide_states[hash(tuple(self.tensor_adapter.to_list(gen_ids[:-1])))]
                curr_state = self.guide.get_next_state(prev_state, self.tensor_adapter.to_list(gen_ids[-1]))
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        mask = self.tensor_adapter.full_like(logits, -math.inf)
        for i, guide_state in enumerate(sequence_states):
            finite_mask = torch.isfinite(logits[i])            # True where logit != -inf
            if not finite_mask.any():                          # all blocked already
                continue
            candidate_ids = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)
            #sorted_candidate_ids = candidate_ids[logits[i, candidate_ids].argsort(descending=True)] 
            sorted_candidate_ids = self.tensor_adapter.argsort_descending(logits[i, candidate_ids])
            valid_ids = list(self.guide.iter_valid_token_ids(guide_state, sorted_candidate_ids))
            if not valid_ids:       # no candidate survived
                continue

            valid_ids = torch.tensor(valid_ids, device=logits.device, dtype=torch.long)
            mask[i, valid_ids] = logits[i, valid_ids]

        return mask