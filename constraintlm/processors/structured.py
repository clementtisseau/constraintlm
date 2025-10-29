import torch
import math
import re
from typing import List

from outlines.processors import OutlinesLogitsProcessor, GuideLogitsProcessor, RegexLogitsProcessor
from .guide import CLMCFGGuide


class CLMLogitsProcessor(OutlinesLogitsProcessor):
    def __init__(self, tensor_library_name: str):
        super().__init__(tensor_library_name=tensor_library_name)


class RPNFullSyntaxLogitsProcessor(RegexLogitsProcessor):
    # Class attribute (constant)
    _number_or_op = re.compile(r"\d+|[+\-*/]")

    def __init__(self, regex_string, tokenizer_outlines, tensor_library_name: str, tokenizer_transformer):
        super().__init__(regex_string=regex_string, tokenizer=tokenizer_outlines, tensor_library_name=tensor_library_name)
        self.tokenizer_transformer = tokenizer_transformer

    def _extract_symbols(self, text: str):
        """
        Parse `text` into a list of RPN tokens (integers and +, -, *, /).
        Returns None if any invalid characters are present.
        """

        matches = list(self._number_or_op.finditer(text))    # list of re.Match objects (it contains the substring, the position of the beginning and the end of the substring)
        symbols = [m.group(0) for m in matches]         # list of substrings that match \d or [+\-*/]
        cleaned = self._number_or_op.sub("", text)           # remove the match from text, we should have "   " only
        if cleaned and not cleaned.isspace():           # we should obtain only whitespace; otherwise text contained a non-digit-nor-operator char
            return None
        return symbols

    def _score_rpn_prefix(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Given the token IDs generated so far and a candidate token (input_ids), 
        return the score associated by the constraint.
        """
        # Decode all sequences at once

        last_id = int(input_ids[-1].item())
        is_eos = last_id == int(self.tokenizer_transformer.eos_token_id) 

        text = self.tokenizer_transformer.decode(input_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        if not text.strip():
            return float("-inf") if is_eos else 0.0      # empty context is valid as a prefix but not a complete sentence
        symbols = self._extract_symbols(text)
        if symbols is None:             # text contains non-digit-nor-operator char => symbols = None 
            return float('-inf')
        depth = 0
        for sym in symbols:
            if sym.isdigit():
                depth += 1
            else:
                # operator
                if depth < 2:
                    return float('-inf')
                depth -= 1
        if is_eos:
            return 0.0 if depth == 1 else float("-inf")
        return 0.0 if depth >= 1 else float("-inf")

    def process_logits(self, input_ids, logits):
        if input_ids.dim() != 2:
            raise ValueError("`input_ids` must be 2-D (batch, seq_len).")
        if logits.dim() != 2:
            raise ValueError("`logits` must be 2-D (batch, vocab).")
        
        processed_logits = super().process_logits(input_ids, logits)    # (B, V)
        # On the first call, this should set the value of self._seq_start_idx = len(input_ids[0])
        
        # VERY important: self._seq_start_idx needs to be the same for all the sentences in the batch
        # So during training we need to use a left-padding
        # And during inference we cannot generate in parallel for different prompts (except if we left-pad) (but we can generate several samples of the same prompt).
        ans_input_ids = input_ids[:, self._seq_start_idx:]
        # print("Answer so far:", self.tokenizer_transformer.batch_decode(ans_input_ids))
        
        if processed_logits.dim() != 2:
            raise ValueError("This processor expects 2-D logits (batch, vocab).")
        
        # for b in range(biased_logits.size(0)):
        #     valid_ids = (biased_logits[b] != float('-inf')).nonzero(as_tuple=True)[0].tolist()
        #     print(f"First masking (batch {b}):", self.tokenizer_transformer.convert_ids_to_tokens(valid_ids))           # for example ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Ġ', 'Ġ*', 'Ġ-', 'Ġ+', 'Ġ/', '<|im_end|>']

        mask = torch.zeros_like(processed_logits, dtype=processed_logits.dtype)

        batch_size = input_ids.size(0)
        for b in range(batch_size):
            seq_prefix = ans_input_ids[b]

            allowed = torch.nonzero(processed_logits[b] != float('-inf'), as_tuple=True)[0]
            for idx in allowed:
                cand_id = idx.to(device=seq_prefix.device, dtype=seq_prefix.dtype).unsqueeze(0)
                cand_seq = torch.cat([seq_prefix, cand_id], dim=0)
                s = self._score_rpn_prefix(cand_seq)
                mask[b, idx] = torch.tensor(s, device=mask.device, dtype=mask.dtype)

        return processed_logits + mask




class CLMCFGLogitsProcessor(GuideLogitsProcessor):        # this is not parallelized at all, could we improve this?  I had to create this new class because the one in outlines was masking everything except the first valid token

    def __init__(self, cfg_str: str, tokenizer, tensor_library_name: str):
        self.cfg_guide = CLMCFGGuide(cfg_string=cfg_str, tokenizer=tokenizer)
        super().__init__(
            tokenizer=tokenizer,
            guide=self.cfg_guide,
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

        def restart(self):
            self._seq_start_idx = None
            self.cfg_guide.initial_state = CFGState(parser_state=self.parser.parse(""), prev_token=None)
            self.cfg_guide.indenter.paren_level = 0