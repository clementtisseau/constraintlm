import re
from typing import List, Optional, Tuple
import torch
from .base import Constraint

# Precompiled regex for matching numbers and operators
_number_or_op_or_var = re.compile(r"\d+|[+\-*/]|[a-zA-Z][a-zA-Z0-9]*")
_var_name = re.compile(r'^[A-Za-z][A-Za-z0-9]*$')

class RPNTypeConstraint(Constraint):
    def __init__(self, llm):
        super().__init__(llm)

    def _extract_symbols(self, text: str) -> Optional[List[str]]:
        """
        Parse `text` into a list of RPN tokens (integers and +, -, *, /).
        Returns None if any invalid characters are present.
        """
        matches = list(_number_or_op_or_var.finditer(text))    # list of re.Match objects (it contains the substring, the position of the beginning and the end of the substring)
        symbols = [m.group(0) for m in matches]         # list of substrings that match \d+ or [+\-*/] or ([a-zA-Z][a-zA-Z0-9])+
        cleaned = _number_or_op_or_var.sub("", text)           # remove the match from text, we should have "   " only
        if cleaned.replace(" ", ""):                    # we should obtain "", otherwise text contained a non-digit-nor-operator char
            return None
        return symbols


    def prefix(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Given the token IDs generated so far (input_ids), 
        return the score associated by the constraint.
        """
        # Decode all sequences at once
        texts = self.llm.tokenizer.batch_decode(
            input_ids, clean_up_tokenization_spaces=False
        )
        scores = []
        
        for text in texts:
            stack = []          # New stack for each sentence
            variables = {}      # New table of variables for each sentence
            
            if not text:
                scores.append(1.0)      # empty context is always a valid prefix
                continue
            symbols = self._extract_symbols(text)
            if symbols is None:             # text contains non-digit-nor-operator char => symbols = None 
                scores.append(0.0)
                continue
            depth = 0
            valid = True
            for sym in symbols:
                if sym.isdigit() or bool(_var_name.match(sym)):
                    depth += 1
                else:
                    # operator
                    if depth < 2:
                        scores.append(0.0)
                        valid = False
                        break
                    depth -= 1
                
                if sym.isdigit():
                    stack.append(float(sym))
                elif bool(_var_name.match(sym)):
                    stack.append(str(sym))
                elif sym == "=":
                    if type(stack[-2]) == float:
                        scores.append(0.0)
                        valid = False
                        break
                    else: # stack[-2] is a string. We never append operators to the stack
                        if type(stack[-1]) == float:    # stack[-1] is a number
                            variables[stack[-2]] = stack[-1]
                            new_value = stack[-1]
                            _ = stack.pop()
                            _ = stack.pop()
                            stack.append(new_value)
                        elif stack[-1] in variables:    # stack[-1] is a defined variable 
                            variables[stack[-2]] = variables[stack[-1]]
                            new_value = variables[stack[-1]]
                            _ = stack.pop()
                            _ = stack.pop()
                            stack.append(new_value)
                        else :                          # stack[-1] is an undefined variable
                            scores.append(0.0)
                            valid = False
                            break
                else:   # any operator that isn't "="
                    if stack[-1] or stack[-2] not in variables: # if there is an undefined variable
                        scores.append(0.0)
                        valid = False
                        break
                    if stack[-1] in variables:                  # if defined, we convert variables into their value
                        stack[-1] = variables[stack[-1]]
                    elif stack[-2] in variables:                # if defined, we convert variables into their value
                        stack[-2] = variables[stack[-2]] 
                    # Now we sould have type(stack[-1]) == float and type(stack[-2]) == float
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(apply_op(a,b, sym))

            if valid:
                scores.append(1.0 if depth >= 1 else 0.0)
        return torch.tensor(scores, dtype=torch.float)


    def complete(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Given the token IDs of a complete (EOS-terminated) sequence (input_ids), 
        return the score associated by the constraint.
        """
        # Decode all sequences at once
        texts = self.llm.tokenizer.batch_decode(
            input_ids, clean_up_tokenization_spaces=False
        )
        scores = []
        stack = []
        variables = {}
        for text in texts:
            symbols = self._extract_symbols(text)
            if symbols is None:             # text contains non-digit-nor-operator char => symbols = None 
                scores.append(0.0)
                continue
            depth = 0
            valid = True
            for sym in symbols:
                if sym.isdigit() or bool(_var_name.match(sym)):
                    depth += 1
                else:
                    # operator
                    if depth < 2:
                        scores.append(0.0)
                        valid = False
                        break
                    depth -= 1
                
                if sym.isdigit():
                    stack.append(float(sym))
                elif bool(_var_name.match(sym)):
                    stack.append(str(sym))
                elif sym == "=":
                    if type(stack[-2]) == float:
                        valid = False
                        break
                    else: # stack[-2] is a string. We never append operators to the stack
                        if type(stack[-1]) == float:    # stack[-1] is a number
                            variables[stack[-2]] = stack[-1]
                            new_value = stack[-1]
                            _ = stack.pop()
                            _ = stack.pop()
                            stack.append(new_value)
                        elif stack[-1] in variables:    # stack[-1] is a defined variable 
                            variables[stack[-2]] = variables[stack[-1]]
                            new_value = variables[stack[-1]]
                            _ = stack.pop()
                            _ = stack.pop()
                            stack.append(new_value)
                        else :                          # stack[-1] is an undefined variable
                            valid = False
                            break
                else:   # any operator that isn't "="
                    if stack[-1] or stack[-2] not in variables: # if there is an undefined variable
                        valid = False
                        break
                    if stack[-1] in variables:                  # if defined, we convert variables into their value
                        stack[-1] = variables[stack[-1]]
                    elif stack[-2] in variables:                # if defined, we convert variables into their value
                        stack[-2] = variables[stack[-2]] 
                    # Now we sould have type(stack[-1]) == float and type(stack[-2]) == float
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(apply_op(a,b, sym))

            if valid:
                scores.append(1.0 if depth == 1 else 0.0)
        return torch.tensor(scores, dtype=torch.float)


    def score(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        For each sequence in `input_ids`, apply `complete` if it ends with EOS, else `prefix`.
        Vectorized selection via torch.where; no explicit Python loop over batch.
        """
        B, L = input_ids.shape

        # Boolean mask of which sequences end in EOS
        is_eos = input_ids[:, -1] == self.llm.eos_token_id  # (B,)
        idx_eos, idx_pref = torch.nonzero(is_eos, as_tuple=True)[0], torch.nonzero(~is_eos, as_tuple=True)[0]   # indices of sequences

        # Prepare output: one score per sequence
        out = torch.empty(B, device=input_ids.device, dtype=torch.get_default_dtype())

        # Score the non-EOS prefixes
        if idx_pref.numel() > 0:
            pref_inputs  = input_ids[idx_pref]          # (Np, L)
            pref_scores  = self.prefix(pref_inputs)     # (Np,) 
            out[idx_pref] = pref_scores

        # Score the EOS-terminated sequences
        if idx_eos.numel() > 0:
            eos_inputs   = input_ids[idx_eos]           # (Ne, L)
            eos_scores   = self.complete(eos_inputs)    # (Ne,) 
            out[idx_eos] = eos_scores

        return out  # shape: (B,)




    def apply(
        self,
        input_ids: Optional[torch.LongTensor],
        probs: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Given the token IDs generated so far (input_ids)
        and the raw next-token probs, return modified probs
        where any token that would violate the RPN constraint
        is masked out (set to zero). Tokens already at zero
        in `probs` remain zero.
        """
        print(input_ids)
        device = probs.device
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        B = probs.size(0)
        assert not torch.any(probs.sum(-1) == 0), "Probabilities sum to 0 before apply()."

        # # handle empty context: if no prefix, everything with non-zero prob is allowed
        # if input_ids is None or input_ids.numel() == 0:
        #     normalizing_cst = torch.ones(probs.shape[0], device=probs.device)
        #     return probs.clone(), normalizing_cst  # nothing to mask beyond the zeros already there

        if input_ids is None or input_ids.numel() == 0:
            input_ids = torch.empty((B, 0), dtype=torch.long, device=device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (1, seqlen)

        bsz = probs.size(0)
        mask = torch.zeros_like(probs)

        for b in range(bsz):
            # find only the tokens with non‐zero probability (not masked)
            candidate_tokens = torch.nonzero(probs[b] > 0, as_tuple=True)[0]
            if candidate_tokens.numel() == 0:           # do we really want to do continue ?
                print(f"for batch {b}, all tokens are already masked")
                continue

            # build a little batch of (prefix + one candidate) for each candidate
            seq = input_ids[b]                  # shape: (seqlen,)
            reps = candidate_tokens.size(0)     # how many candidates
            seqs = seq.unsqueeze(0).repeat(reps, 1)            # (reps, seqlen)
            next_ids = candidate_tokens.unsqueeze(1)           # (reps, 1)
            batch = torch.cat([seqs, next_ids], dim=1)         # (reps, seqlen+1)

            scores = self.score(batch)          # tensor of 0.0 or 1.0, shape (reps,)
            assert not torch.any(scores.sum(-1) == 0), f"scores for batch {b} sum to 0."

            mask[b, candidate_tokens] = scores

        # zero out forbidden tokens, keep the rest of the distribution as-is
        unnormalized_probs = probs * mask                 # (B, V)
        normalizing_cst = unnormalized_probs.sum(-1)    #(B)
        assert not torch.any(unnormalized_probs.sum(-1) == 0), "Probabilities sum to 0 after mask."


        normalized_probs = unnormalized_probs / (normalizing_cst.unsqueeze(1) + 1e-8)

        return normalized_probs, normalizing_cst



def apply_op(a: float, b: float, op: str) -> float:
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b
    else:
        raise ValueError(f"Unsupported operator {op!r}")