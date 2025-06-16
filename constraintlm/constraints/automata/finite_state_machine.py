from collections import defaultdict, deque
from typing import Dict, Set, Tuple, Optional, FrozenSet
import sre_parse
import string

EPSILON = ""
ANY_SYMBOL = None

# Is the DFA a minimal DFA ? Is the transition function total in this case ? Thus does it include a death state ?

class FiniteStateMachine:
    """
    Very lightweight representation of a finite‑state machine.
    Transitions may include "" for ""‑moves, allowing the same class to
    represent both NFAs and DFAs.  A deterministic machine is recovered with
    the `to_dfa` method.
    """


    def __init__(self, states: Optional[Set[int]] = None, alphabet: Optional[Set[str]] = None, transitions: Optional[Dict[Tuple[int, str], Set[int]]] = None, start_state: Optional[int] = None, accept_states: Optional[Set[int]] = None):
        self.states: Set[int] = states if states is not None else set()
        # alphabet is the set of char explicitly written in the regex/FSM (when we have "." we include all of them in alphabet)
        self.alphabet: Set[str] = alphabet if alphabet is not None else set()
        self.transitions: Dict[Tuple[int, str], Set[int]] = transitions if transitions is not None else defaultdict(set)
        self.start_state: Optional[int] = start_state
        self.accept_states: Set[int] = accept_states if accept_states is not None else set()

    # --------------------------------------------------------------
    # Basic construction helpers
    # --------------------------------------------------------------
    def add_state(self, *, is_start: bool = False, is_accept: bool = False) -> int:
        """Create a new state and return its id."""
        new_id = len(self.states)
        self.states.add(new_id)
        if is_start:
            self.start_state = new_id
        if is_accept:
            self.accept_states.add(new_id)
        return new_id

    def add_transition(self, src: int, symbol: str, dst: int) -> None:
        """Add a transition `src --symbol--> dst`."""
        if symbol != EPSILON:
            self.alphabet.add(symbol)
        self.transitions[(src, symbol)].add(dst)

    # --------------------------------------------------------------
    # Execution
    # --------------------------------------------------------------
    def _epsilon_closure(self, states: Set[int]) -> Set[int]:
        """Return the ""‑closure of *states*."""
        stack = list(states)
        closure = set(states)
        while stack:
            s = stack.pop()
            for nxt in self.transitions.get((s, EPSILON), ()):  # returns empty tuple as default if (state, ch) is not a key in the dict
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)
        return closure

    def is_accepting(self, s: str) -> bool:
        """True iff the machine accepts the string *s*."""
        current = self._epsilon_closure({self.start_state})
        for ch in s:
            nxt: Set[int] = set()
            for state in current:
                # We add all states than can be reached after reading ch
                for dest in self.transitions.get((state, ch), ()):
                    nxt.update(self._epsilon_closure({dest}))
                if ch != "\n":  # we don't want "a.b" to read "a\nb"
                    # We add all states than can be reached after reading any symbol
                    for dest in self.transitions.get((state, ANY_SYMBOL), ()):
                        nxt.update(self._epsilon_closure({dest}))
            current = nxt
            if not current:
                return False
        return bool(self.accept_states & current)

    # --------------------------------------------------------------
    # NFA → DFA
    # --------------------------------------------------------------
    def to_dfa(self) -> "FiniteStateMachine":
        """Determinise the machine with the subset construction."""
        start = frozenset(self._epsilon_closure({self.start_state}))    # immutable version of a set
        mapping: Dict[FrozenSet[int], int] = {}
        dfa = FiniteStateMachine()
        queue: deque[FrozenSet[int]] = deque()

        def state_id(subset: FrozenSet[int]) -> int:
            if subset not in mapping:
                mapping[subset] = dfa.add_state(
                    is_start=not mapping,
                    is_accept=bool(self.accept_states & subset),
                )
                queue.append(subset)
            return mapping[subset]

        state_id(start)

        while queue:
            subset = queue.popleft()
            dfa_src = mapping[subset]

            # --- (A) Handle wildcard‐moves (ANY_SYMBOL) ---
            target_any: Set[int] = set()
            for st in subset:
                # collect any NFA‐states reachable by an (st, ANY_SYMBOL) arc
                target_any.update(self.transitions.get((st, ANY_SYMBOL), ()))
            if target_any:
                closure_any = self._epsilon_closure(target_any)
                if closure_any:
                    dfa_dest = state_id(frozenset(closure_any))
                    # record one DFA‐edge (dfa_src, ANY_SYMBOL) → dfa_dest
                    dfa.add_transition(dfa_src, ANY_SYMBOL, dfa_dest)

            # --- (B) Handle each concrete symbol in self.alphabet ---
            for symbol in self.alphabet:
                target: Set[int] = set()
                for st in subset:
                    # follow any (st, symbol) arcs in the NFA
                    target.update(self.transitions.get((st, symbol), ()))
                if not target:
                    continue
                closure = self._epsilon_closure(target)
                if not closure:
                    continue
                dfa_dest = state_id(frozenset(closure))
                # record DFA‐edge (dfa_src, symbol) → dfa_dest
                dfa.add_transition(dfa_src, symbol, dfa_dest)


        print(self.transitions)
        print(dfa.transitions)
        return dfa

    # --------------------------------------------------------------
    # Debug helpers
    # --------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FiniteStateMachine(start={self.start_state}, "
            f"accept={self.accept_states}, "
            f"alphabet={self.alphabet}, "
            f"transitions={dict(self.transitions)})"
        )

    @classmethod
    def from_regex(cls, pattern: str) -> "FiniteStateMachine":
        """
        Parse a Python‐re pattern (restricted to the purely regular subset)
        and build an NFA via Thompson’s construction.
        """
        # 1) Parse with the builtin engine
        parsed = sre_parse.parse(pattern)
        print(parsed)

        # 2) Prepare an explicit, finite alphabet by walking the parse tree
        def collect_alphabet(parsed) -> Set[str]:
            alpha: Set[str] = set()
            for op, arg in parsed:
                if op is sre_parse.LITERAL:     # match exactly one specific character
                    alpha.add(chr(arg))
                elif op is sre_parse.ANY:
                    # wildcard: we’ll expand later over alpha
                    pass
                elif op is sre_parse.IN:    # bracketed character‐class like [a-z0-9_]
                                            # arg is a list of (subop, subarg)
                    for subop, subarg in arg:
                        if subop is sre_parse.LITERAL:  # match exactly one specific character
                            alpha.add(chr(subarg))
                        elif subop is sre_parse.RANGE:
                            start, end = subarg
                            alpha.update(chr(c) for c in range(start, end+1))
                        elif subop is sre_parse.CATEGORY:
                            if subarg is sre_parse.CATEGORY_DIGIT:
                                alpha.update(string.digits)
                            elif subarg is sre_parse.CATEGORY_WORD:
                                alpha.update(string.ascii_letters + string.digits + "_")
                            elif subarg is sre_parse.CATEGORY_SPACE:
                                alpha.update(" \t\n\r\f\v")
                            else:
                                raise ValueError(f"Unsupported category in class: {subarg}")
                        else:
                            raise ValueError(f"Unsupported token in class: {subop}")
                elif op is sre_parse.SUBPATTERN:
                    subparsed = arg[3]
                    alpha |= collect_alphabet(subparsed)
                elif op is sre_parse.BRANCH:
                    _, branches = arg
                    for b in branches:
                        alpha |= collect_alphabet(b)
                elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                    lo, hi, subparsed = arg
                    alpha |= collect_alphabet(subparsed)
                elif op is sre_parse.CATEGORY:
                    # top‐level \d,\w,\s
                    if arg is sre_parse.CATEGORY_DIGIT:
                        alpha.update(string.digits)
                    elif arg is sre_parse.CATEGORY_WORD:
                        alpha.update(string.ascii_letters + string.digits + "_")
                    elif arg is sre_parse.CATEGORY_SPACE:
                        alpha.update(" \t\n\r\f\v")
                    else:
                        raise ValueError(f"Unsupported category: {arg}")
                else:
                    raise ValueError(f"Unsupported or non-regular token: {op}")
            return alpha

        alphabet = collect_alphabet(parsed)

        # 3) Thompson fragment
        class Frag:
            __slots__ = ('start','accept','trans')
            def __init__(self, start:int, accept:Set[int], trans:Dict[Tuple[int,str],Set[int]]):
                self.start, self.accept, self.trans = start, accept, trans

        next_id = 0
        def new_state() -> int:
            nonlocal next_id
            s = next_id
            next_id += 1
            return s

        # 4) Compile parse-tree → Frag
        def compile_subparsed(parsed) -> Frag:
            """Compile a list of (op,arg) into one concatenated Frag."""
            if list(parsed) == []:    # parsed is empty iff pattern is ""
                return Frag(start=0, accept={0}, trans=None)
            else: 
                frags = []
                for op, arg in parsed:
                    if op is sre_parse.LITERAL:
                        s0, s1 = new_state(), new_state()
                        t = {(s0, chr(arg)): {s1}}
                        frags.append(Frag(s0, {s1}, t))

                    elif op is sre_parse.ANY:
                        s0, s1 = new_state(), new_state()
                        t = {(s0, ANY_SYMBOL): {s1}}
                        frags.append( Frag(s0, {s1}, t))

                    elif op is sre_parse.IN:
                        # class: union of literals/ranges/categories
                        # we build one frag with s0→s1 for each member
                        members: Set[str] = set()
                        for subop, subarg in arg:
                            if subop is sre_parse.LITERAL:
                                members.add(chr(subarg))
                            elif subop is sre_parse.RANGE:
                                members |= {chr(c) for c in range(subarg[0], subarg[1]+1)}
                            elif subop is sre_parse.CATEGORY:
                                if subarg is sre_parse.CATEGORY_DIGIT:
                                    members |= set(string.digits)
                                elif subarg is sre_parse.CATEGORY_WORD:
                                    members |= set(string.ascii_letters + string.digits + "_")
                                elif subarg is sre_parse.CATEGORY_SPACE:
                                    members |= set(" \t\n\r\f\v")
                            else:
                                raise ValueError(f"Unsupported in-class op {subop}")
                        s0, s1 = new_state(), new_state()
                        t = {}
                        for m in members:
                            t.setdefault((s0, m), set()).add(s1)
                        frags.append(Frag(s0, {s1}, t))

                    elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                        lo, hi, subseq = arg
                        # lo: int                               lower bound
                        # hi : int or sre_parse.MAX_REPEAT      upper bound (could be infinite -> hi = sre_parse.MAX_REPEAT)
                        # subseq : List[(ops, args)]
                        base = compile_subparsed(subseq)

                        # k repetitions
                        def repeat(f: Frag, k: int) -> Frag:
                            # Helper to clone a Frag into brand-new state IDs
                            def clone_fragment(f: Frag) -> Frag:
                                # 1) Collect all old states
                                old_states: Set[int] = set()
                                for (src, _), dsts in f.trans.items():
                                    old_states.add(src)
                                    old_states.update(dsts)
                                old_states |= f.accept
                                old_states.add(f.start)

                                # 2) Create a mapping to new states
                                mapping: Dict[int, int] = {}
                                for old in old_states:
                                    mapping[old] = new_state()

                                # 3) Build new transitions
                                new_trans: Dict[Tuple[int, str], Set[int]] = {}
                                for (old_src, sym), old_dsts in f.trans.items():
                                    new_src = mapping[old_src]
                                    new_trans.setdefault((new_src, sym), set())
                                    for old_dst in old_dsts:
                                        new_trans[(new_src, sym)].add(mapping[old_dst])

                                # 4) Compute new start and accept
                                new_start = mapping[f.start]
                                new_accept = {mapping[a] for a in f.accept}
                                return Frag(new_start, new_accept, new_trans)

                            # If k == 0, return an ""-only fragment
                            if k == 0:
                                s = new_state()
                                return Frag(s, {s}, {})

                            # 1st copy
                            result_frag = clone_fragment(base)

                            # For each additional repetition, concatenate a fresh clone
                            for _ in range(1, k):
                                next_copy = clone_fragment(base)
                                # Link each accept of result_frag → "" → next_copy.start
                                merged_trans = {**result_frag.trans}
                                for a in result_frag.accept:
                                    merged_trans.setdefault((a, EPSILON), set()).add(next_copy.start)
                                # Merge next_copy's transitions
                                for (src, sym), dsts in next_copy.trans.items():
                                    merged_trans.setdefault((src, sym), set()).update(dsts)
                                result_frag = Frag(result_frag.start, set(next_copy.accept), merged_trans)

                            return result_frag

                        # unbounded star for the "infinite" tail
                        if hi == sre_parse.MAXREPEAT:
                            # This is either *, +, or {m,}
                            if lo == 0:     # * 
                                frags.append(star(base))
                            elif lo == 1:   # +
                                frags.append(plus(base))
                            else:           # a{m,} = (a · a · … · a) · a*
                                frags.append(concat(repeat(base, lo), star(base)))
                        else:
                            # finite {lo, hi}
                            if lo == hi:    # e.q. a{4}
                                frags.append(repeat(base, lo))
                            else:           # e.q. a{2,6}
                            # {lo,hi} with hi>lo
                            # build union of k in [lo..hi]
                                options = [repeat(base, k) for k in range(lo, hi+1)]
                                frags.append(union(options))

                    elif op is sre_parse.SUBPATTERN:
                        subparsed = arg[3]
                        frags.append(compile_subparsed(subparsed))

                    elif op is sre_parse.BRANCH:
                        # arg = (None, [seq1, seq2, …])
                        branches = arg[1]
                        branch_frags = [compile_subparsed(b) for b in branches]
                        frags.append(union(branch_frags))

                    elif op is sre_parse.CATEGORY:
                        # top-level \d,\w,\s
                        if arg is sre_parse.CATEGORY_DIGIT:
                            frags.append(compile_subparsed([ (sre_parse.IN, [(sre_parse.RANGE, (ord('0'),ord('9')))]) ]))
                        elif arg is sre_parse.CATEGORY_WORD:
                            rngs = list(string.ascii_letters + string.digits + "_")
                            frags.append(compile_subparsed([ (sre_parse.IN, [(sre_parse.LITERAL, ord(c)) for c in rngs]) ]))
                        elif arg is sre_parse.CATEGORY_SPACE:   # \s
                            frags.append(compile_subparsed([ (sre_parse.IN, [(sre_parse.LITERAL, ord(c)) for c in " \t\n\r\f\v"]) ]))
                        else:
                            raise ValueError(f"Unsupported category {arg}")

                    else:
                        raise ValueError(f"Unsupported or non-regular token: {op}")

                # concatenate all fragments in `frags`
                res = frags[0]
                for f in frags[1:]:
                    res = concat(res, f)
                return res
    

        # 5) Thompson helpers
        def concat(f1: Frag, f2: Frag) -> Frag:
            trans = {**f1.trans}
            for a in f1.accept:
                trans.setdefault((a, EPSILON), set()).add(f2.start)
            trans.update(f2.trans)
            return Frag(f1.start, set(f2.accept), trans)

        def union(frags: list[Frag]) -> Frag:
            s0, s1 = new_state(), new_state()
            trans = {}
            for f in frags:
                # "" → each start
                trans.setdefault((s0, EPSILON), set()).add(f.start)     # add() for one element
                # merge each frag’s transitions
                for (src, sym), dsts in f.trans.items():
                    trans.setdefault((src, sym), set()).update(dsts)    # update() for an iterable
                # each accept → "" → s1
                for a in f.accept:
                    trans.setdefault((a, EPSILON), set()).add(s1)
            return Frag(s0, {s1}, trans)

        def star(f: Frag) -> Frag:
            s0, s1 = new_state(), new_state()
            trans = {**f.trans} # copy the dict
            # s0 → "" → f.start & s0 → "" → s1
            trans.setdefault((s0, EPSILON), set()).update({f.start, s1})
            for a in f.accept:
                # each accept → "" → f.start & "" → s1
                trans.setdefault((a, EPSILON), set()).update({f.start, s1})
            return Frag(s0, {s1}, trans)

        def plus(f: Frag) -> Frag:
            # one copy followed by star
            return concat(f, star(f))

        # 6) Build the NFA frag 
        frag = compile_subparsed(parsed)

        return cls(states=set(range(next_id)), alphabet=alphabet, transitions=frag.trans, start_state=frag.start, accept_states=set(frag.accept))


# ----- Old version that doesn't support any meta-characters -----

# ------------------------------------------------------------------
# Regex → NFA (Thompson, with shunting‑yard for parsing)
# ------------------------------------------------------------------

# def _insert_concat(regex: str) -> str:
#     """Insert explicit concatenation operator '.' where implied."""
#     out: list[str] = []
#     for i, c in enumerate(regex):
#         if c == ' ':  # ignore whitespace
#             continue
#         if (
#             i   #non-zero
#             and regex[i - 1] not in '(|'
#             and c not in '|)*+?)'
#         ):
#             out.append('.')
#         out.append(c)
#     return ''.join(out)


# def _to_postfix(regex: str) -> str:
#     prec = {'*': 3, '+': 3, '?': 3, '.': 2, '|': 1}
#     output: list[str] = []
#     stack: list[str] = []
#     for c in regex:
#         if c == '(':
#             stack.append(c)
#         elif c == ')':
#             while stack and stack[-1] != '(':
#                 output.append(stack.pop())
#             stack.pop()
#         elif c in prec:
#             while stack and stack[-1] != '(' and prec[stack[-1]] >= prec[c]:
#                 output.append(stack.pop())
#             stack.append(c)
#         else:
#             output.append(c)
#     while stack:
#         output.append(stack.pop())
#     return ''.join(output)

# def from_regex(pattern: str) -> FiniteStateMachine:
#     """Convert *pattern* (supports |, *, +, ?, (, )) to an NFA FSM."""
#     pattern = _insert_concat(pattern)
#     postfix = _to_postfix(pattern)

#     class Frag:
#         __slots__ = ('start', 'accept', 'trans')
#         def __init__(self, start: int, accept: Set[int], trans: Dict[Tuple[int, str], Set[int]]):
#             self.start, self.accept, self.trans = start, accept, trans

#     next_state = 0

#     def new_state() -> int:
#         nonlocal next_state
#         s = next_state
#         next_state += 1
#         return s

#     stack: list[Frag] = []

#     for tok in postfix:
#         if tok not in '|*+?.':
#             s0, s1 = new_state(), new_state()
#             trans = {(s0, tok): {s1}}
#             stack.append(Frag(s0, {s1}, trans))
#         elif tok == '.':
#             f2 = stack.pop()
#             f1 = stack.pop()
#             trans = {**f1.trans, **f2.trans}
#             for a in f1.accept:
#                 trans.setdefault((a, EPSILON), set()).add(f2.start)
#             stack.append(Frag(f1.start, f2.accept, trans))
#         elif tok == '|':
#             f2 = stack.pop()
#             f1 = stack.pop()
#             s0, s1 = new_state(), new_state()
#             trans = {**f1.trans, **f2.trans}
#             trans[(s0, EPSILON)] = {f1.start, f2.start}
#             for a in f1.accept | f2.accept:
#                 trans.setdefault((a, EPSILON), set()).add(s1)
#             stack.append(Frag(s0, {s1}, trans))
#         elif tok == '*':
#             f = stack.pop()
#             s0, s1 = new_state(), new_state()
#             trans = {**f.trans}
#             trans[(s0, EPSILON)] = {f.start, s1}
#             for a in f.accept:
#                 trans.setdefault((a, EPSILON), set()).update({f.start, s1})
#             stack.append(Frag(s0, {s1}, trans))
#         elif tok == '+':
#             f = stack.pop()
#             s0, s1 = new_state(), new_state()
#             trans = {**f.trans}
#             trans[(s0, EPSILON)] = {f.start}
#             for a in f.accept:
#                 trans.setdefault((a, EPSILON), set()).update({f.start, s1})
#             stack.append(Frag(s0, {s1}, trans))
#         elif tok == '?':
#             f = stack.pop()
#             s0, s1 = new_state(), new_state()
#             trans = {**f.trans}
#             trans[(s0, EPSILON)] = {f.start, s1}
#             for a in f.accept:
#                 trans.setdefault((a, EPSILON), set()).add(s1)
#             stack.append(Frag(s0, {s1}, trans))

#     if len(stack) != 1:
#         raise ValueError("Malformed pattern")

#     frag = stack.pop()
#     fsm = FiniteStateMachine()
#     fsm.states = set(range(next_state))
#     fsm.start_state = frag.start
#     fsm.accept_states = set(frag.accept)
#     for (src, sym), dsts in frag.trans.items():
#         for dst in dsts:
#             fsm.add_transition(src, sym, dst)
#     return fsm



# ------------------------------------------------------------------
# Quick demonstration
# ------------------------------------------------------------------
if __name__ == "__main__":
    pattern = "(a|b)*abb"
    nfa = FiniteStateMachine.from_regex(pattern)
    dfa = nfa.to_dfa()
    print(f"accepts 'abb'?  {dfa.is_accepting('abb')}")
    print(f"accepts 'aabb'? {dfa.is_accepting('aabb')}")
    print(f"accepts 'ab'?   {dfa.is_accepting('ab')}")