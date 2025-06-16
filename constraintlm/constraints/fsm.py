import re
import torch
from .base import Constraint
from .automata.finite_state_machine import FiniteStateMachine
from typing import Tuple

# ------------ I need to make all functions accepintg batches. current_state must be a tensor of shape (B). 

# A FSMConstraint, contrary to other constraints so far, have parameters (self.current_state) that evolve during the generation (at each call to apply()). 
# It means that when we create such a constraint, it is usable only one time, or the self.current_state must be reinitialized.

# Is FiniteStateMachine from_regex().to_dfa() creating a death_state ? No
# Could this be a problem ? It means that the function \delta isn't defined rigorously.
# function to create: find_sub_sequence(), process_string()

class FSMConstraint(Constraint):
    """
    Only allows next tokens that keep the generated text matching the given regex.
    """
    def __init__(self, llm, fsm: FiniteStateMachine):
        super().__init__(llm)
        #self.vocab = llm.tokenizer.get_vocab()  # dict ("token" -> token_id)
        self.fsm = fsm


        # Dict that maps every state to a Dict of (acceptable_token_id -> next state)
        self._transitions = None            # = {"state_0": {0:  "state_5", 14: "state_2", 28: "state_9", }, "state_1": {2:  "state_3", 7:  "state_8", 11: "state_4", }, …}
        # Dict that maps every state to a list of acceptable tokens # This one is optional, we can have it by doing list(transitions["state_0"].keys())  # [0, 14, 28]
        self._acceptable_token_ids = None   # = {"state_0": [0, 14, 28],"state_1": [2, 7, 11], …}

        self.current_state = self.fsm.start_state



    def apply(self, input_ids: torch.LongTensor, probs: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        device = probs.device
        B = probs.size(0)      # Assume input_ids/probs has necessarly been flatten at this point # input_ids : (B, t) with t the n° of generated tokens so far
        if input_ids is None or input_ids.numel() == 0:
            input_ids = torch.empty((B, 0), dtype=torch.long, device=device)

        if self.current_state == self.fsm.start_state:      # First call to apply()
            self.current_state = [self.fsm.start_state] * B
        
        if input_ids.size(1) > 0:
            last_tokens = input_ids[:, -1]      # (B,)
            self.current_state = [self._update_current_state(last_tokens[b].item(), self.current_state[b]) for b in range(B)]
        
        acceptable_next_token_ids = [self.acceptable_token_ids[s] for s in self.current_state]     # list of lists
        mask = torch.zeros((B, self.llm.logits_size), device=device)
        for b, token_list in enumerate(acceptable_next_token_ids):
            mask[b, token_list] = 1.0
        unnormalized_probs = probs * mask                 # (B, V)
        normalizing_cst = unnormalized_probs.sum(-1)    #(B)

        normalized_probs = unnormalized_probs / (normalizing_cst.unsqueeze(1) + 1e-8)
        assert not torch.any(torch.isinf(normalized_probs)), "Probabilities contain inf values."
        assert not torch.any(torch.isnan(normalized_probs)), "Probabilities contain nan values."
        assert torch.all(normalized_probs >= 0), "Probabilities contain negative values."
        
        assert not torch.any(torch.isinf(normalizing_cst)), "Normalizing constant contains inf values."
        assert not torch.any(torch.isnan(normalizing_cst)), "Normalizing constant contains nan values."
        assert torch.all(normalizing_cst >= 0), "Normalizing constant contains negative values."
        assert not torch.any(normalizing_cst == 0), "Probabilities sum to 0."

        return normalized_probs, normalizing_cst
        

    def _update_current_state(self, last_token_id, current_state):
        # updated_state = self.transitions[current_state][last_token_id]   # 
        # return updated_state
        trans = self.transitions[current_state]
        if last_token_id not in trans:
            print(f"No transition from state {current_state} on token {last_token_id}")
            print("Allowed tokens here:", list(trans.keys()))
            raise KeyError(last_token_id)
        return trans[last_token_id]

    def create_hash_tables(self):
        # create the hash tables 
        self._transitions = {state: {} for state in self.fsm.states}

        for token_id in range(self.llm.vocab_token_size): # doesn't include added tokens (no eos_token, pad_token, etc.)
            sub_sequences = self.find_sub_sequences(token_id)
            for ss in sub_sequences:
                future_state = self.process_string(ss[0], self.llm.tokenizer.decode(token_id))   # None if the string cannot be processed
                if future_state is not None:
                    self._transitions[ss[0]][token_id] = future_state
        for state_a in self.fsm.accept_states:
            self._transitions[state_a][self.llm.eos_token_id] = state_a

        self._acceptable_token_ids = {state: list(self._transitions[state].keys()) for state in self._transitions.keys()}

    def find_sub_sequences(self, token_id: int):
        # Given a token_id, finds all sub_sequences of states that read the token (starting from any state). 
        string_token = self.llm.tokenizer.decode(token_id)
        sub_sequences = []

        for q in self.fsm.states:
            dests = self.fsm.transitions.get((q, string_token[0]))
            if not dests:   # if dests is None: go to next iteration
                continue
            ss = [q]
            current = next(iter(dests))     # We need the FSM to be a DFA so dests contains at most 1 element

            for c in string_token[1:]:
                dests = self.fsm.transitions.get((current, c))    # if the tuple (state, char) doesn't have a value, it returns None
                if not dests:   # if dests is None: exit this small for loop
                    break
                ss.append(current)
                current = next(iter(dests))
            else: # ONLY runs if we never broke out of the loop, it belongs to for
                sub_sequences.append(ss)
        return sub_sequences


    def process_string(self, state: int, s: str):
        current = state
        for c in s:
            if c not in self.fsm.alphabet:      # any char not explicitly defined in the regex/FSM
                c = None
            dests = self.fsm.transitions.get((current, c))      #transitions[(state, None)] is the destination (unique if is a DFA) after reading anything not in the alphabet, from state. 
            if not dests:   # if dests is None (no transition on this symbol: dead / undefined)
                return None
            current = next(iter(dests))
        return current


    @property
    def transitions(self):
        if self._transitions is None:
            raise RuntimeError("transitions is not built; call create_hash_tables() first")
        return self._transitions
    
    @property
    def acceptable_token_ids(self):
        if self._acceptable_token_ids is None:
            raise RuntimeError("acceptable_token_ids is not built; call create_hash_tables() first")
        return self._acceptable_token_ids