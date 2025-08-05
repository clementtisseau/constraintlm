import collections
import copy
import warnings
from typing import TYPE_CHECKING, Any, Generator, ValuesView, Union

from lark.indenter import DedentError, PostLex
from lark.lexer import UnexpectedCharacters, UnexpectedToken
from outlines_core.fsm.guide import Generate
from outlines_core.fsm.guide import Write
from outlines_core.fsm.guide import (
    create_states_mapping as uncached_create_states_mapping,
)

from outlines import grammars
from outlines.caching import cache
from outlines.fsm.parsing import PartialLark, PartialParserState

if TYPE_CHECKING:
    from outlines.models.tokenizer import Tokenizer


Instruction = Union[Write, Generate]


from outlines.processors.guide import Guide

CFGState = collections.namedtuple("CFGState", ["parser_state", "prev_token"])

PY_KEYWORDS = {
    "False","None","True","and","as","assert","async","await",
    "break","class","continue","def","del","elif","else","except",
    "finally","for","from","global","if","import","in","is","lambda",
    "nonlocal","not","or","pass","raise","return","try",
    "while","with","yield"
}

class KeywordIndenter(PythonIndenter):

    _keyword_tokens = tuple(k.upper() for k in PY_KEYWORDS)

    # override the *property*
    @property
    def always_accept(self):
        return super().always_accept + self._keyword_tokens

    def process(self, stream):
        for tok in super().process(stream):
            if tok.type == "NAME" and tok.value in PY_KEYWORDS:
                tok.type = tok.value.upper()
            yield tok

class CLMCFGGuide(Guide):
    """Guide to generate text that is in the language of a context-free Lark
    grammar.

    """

    def __init__(self, cfg_string: str, tokenizer: "Tokenizer"):
        """
        Parameters
        ----------
        cfg_string
            The context-free grammar to generate text from.
        tokenizer
            The tokenizer to use to convert tokens to ids.

        """
        warnings.warn(
            "Outlines' public *community-contributed* CFG structured generation "
            "is experimental. Please review "
            "https://dottxt-ai.github.io/outlines/latest/reference/generation/cfg#disclaimer"
        )

        self.cfg_string = cfg_string
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.parser = PartialLark(
            cfg_string,
            parser="lalr",
            lexer="contextual",
            postlex = KeywordLocker(),
            import_paths=[grammars.GRAMMAR_PATH],
        )
        self.initial_state = CFGState(
            parser_state=self.parser.parse(""), prev_token=None
        )

    def get_next_instruction(self, state: CFGState) -> Instruction:
        """Return the next instruction for guided generation.

        Current lazy approach:
        - For each token in the vocabulary
          - create a copy of the parsers state
          - add the tokens to the parsers input text
          - if valid, add token to returned tokens

        Further refinements are necessary for performant text processing.

        Parameters
        ----------
        state
            The guides current PartialParserState, or None if complete

        Returns
        -------
        Instruction
            A `Generate` instance that contains the model and the allowed token
            ids.

        """
        import torch

        if state.parser_state is None:
            return Write(torch.tensor([self.eos_token_id]))

        valid_tokens = list(
            self.iter_valid_token_ids(
                state, self.tokenizer.vocabulary.values()
            )
        )

        if len(valid_tokens) == 1:
            return Write(torch.tensor(valid_tokens))

        return Generate(torch.tensor(valid_tokens))

    def iter_valid_token_ids(
        self, state: CFGState, candidate_token_ids: ValuesView[int]
    ) -> Generator[int, None, None]:
        """Iterate over the given token_ids and yield those that are valid for
        the current parser state.

        Parameters
        ----------
        parser_state
            The current state of the parser, or None if complete.
        token_ids
            The list of token ids to check for validity.

        Yields
        ------
        int
            Valid token ids.

        """
        for token_id in candidate_token_ids:
            if token_id == self.eos_token_id:
                if self.can_terminate_state(state):
                    yield token_id
            else:
                try:
                    self._get_parser_state_token_applied(state, int(token_id))
                    yield token_id
                except (
                    ValueError,
                    EOFError,
                    UnexpectedToken,
                    UnexpectedCharacters,
                    DedentError,
                ):
                    pass

    def get_next_state(self, state: CFGState, token_id: int) -> CFGState:
        """Update the state of the guide.

        Decode the token_id, and calculate the new parser_state with the token
        applied.

        Parameters
        ----------
        state
            The guides current PartialParserState, or None if complete
        token_id
            The id of the token that was just generated.

        Returns
        -------
        CFGState
            The guides new PartialParserState

        """
        if state.parser_state is None or token_id == self.eos_token_id:
            parser_state = None
        else:
            parser_state = self._get_parser_state_token_applied(state, int(token_id))
        return CFGState(parser_state=parser_state, prev_token=token_id)

    def _get_parser_state_token_applied(
        self, state: CFGState, token_id: int
    ) -> PartialParserState:
        """Apply the given token_id to the parser state.

        Don't mutate `parser_state`, copy to protect

        Get the token string
          - if first token in generation: tokenizer.decode (no leading whitespace)
          - else: normalized (with possibly leading whitespace)

        Don't allow empty ("") tokens, raise ValueError

        Parameters
        ----------
        state
            The guide's current PartialParserState, or None if complete
        token_id
            The id of the token that was just generated.

        Returns
        -------
        PartialParserState
            The parser state with the token applied.

        """
        parser_state = copy.copy(state.parser_state)  # prevent side effects

        # normalize
        if state.prev_token is None:
            new_token_str = self.tokenizer.decode([token_id])[0]
        else:
            prev_token_str = self.tokenizer.decode([[state.prev_token]])[0]
            combined_token_str = self.tokenizer.decode([[state.prev_token, token_id]])[
                0
            ]
            new_token_str = combined_token_str[len(prev_token_str) :]

        if new_token_str == "":
            raise ValueError("empty next token")

        # update parser with new token
        parser_state.lexer.state.text += new_token_str
        self.parser.parse_from_state(parser_state, is_end=False)

        return parser_state

    def is_final_state(self, state: CFGState) -> bool:
        """Return whether the given state is a final state.

        Parameters
        ----------
        state
            The guide's current state.

        Returns
        -------
        bool
            Whether the given state is a final state.

        """
        # TODO: remove this method, use can_terminate_state and
        # must_terminate_state here and in RegexGuide per
        # https://github.com/dottxt-ai/outlines/issues/885
        return self.can_terminate_state(state)

    def can_terminate_state(self, state: CFGState) -> bool:
        """Return whether generation is allowed to terminate.

        Parameters
        ----------
        state
            The guide's current state.

        Returns
        -------
        bool
            Whether generation is allowed to terminate.

        """
        if state.parser_state is not None:
            try:
                copy.copy(state.parser_state).feed_eof()
            except UnexpectedToken:
                return False
        return True

    def must_terminate_state(self, state: CFGState) -> bool:
        """Indicate whether generation must terminate as there are no legal
        continuations.

        Parameters
        ----------
        state
            The guide's current state.

        Returns
        -------
        bool
            Whether generation must terminate.

        """
        return (
            state.parser_state is None or
            set(state.parser_state.accepts()).issubset({"$END"})
        )

    def copy(self) -> "CLMCFGGuide":
        """Create a copy of the Guide.

        Returns
        -------
        CFGGuide
            A copy of the Guide.

        """
        return CLMCFGGuide(self.cfg_string, self.tokenizer)