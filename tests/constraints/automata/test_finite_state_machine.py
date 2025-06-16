from constraintlm.constraints.automata.finite_state_machine import FiniteStateMachine

def test_simple_literal():
    # "a" should accept exactly "a"
    nfa = FiniteStateMachine.from_regex("a")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("a")
    assert not dfa.is_accepting("")
    assert not dfa.is_accepting("b")
    assert not dfa.is_accepting("aa")

def test_union_and_concat():
    # "(a|b)c" should accept "ac" or "bc" only
    nfa = FiniteStateMachine.from_regex("(a|b)c")
    print(nfa)
    dfa = nfa.to_dfa()
    print(dfa)
    assert dfa.is_accepting("ac")
    assert dfa.is_accepting("bc")
    assert not dfa.is_accepting("c")
    assert not dfa.is_accepting("abc")

def test_kleene_star():
    nfa = FiniteStateMachine.from_regex("(a|b)*abb")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("abb")   
    assert dfa.is_accepting("aabb")  
    assert dfa.is_accepting("babb")  
    assert not dfa.is_accepting("ab")   # too short
    assert not dfa.is_accepting("")     # empty string

def test_plus():
    nfa = FiniteStateMachine.from_regex("(a|b)+abb")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("aabb")  
    assert dfa.is_accepting("babb")  
    assert not dfa.is_accepting("abb")  # too short
    assert not dfa.is_accepting("ab")   # too short
    assert not dfa.is_accepting("")     # empty string

def test_character_classes_and_quantifiers():
    nfa = FiniteStateMachine.from_regex(r"\d{3}-\d{2}")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("123-45")
    assert not dfa.is_accepting(r"\d{3}-\d{2}") # check if it understands meta-characters
    assert not dfa.is_accepting("12-345")
    assert not dfa.is_accepting("1234-56")
    assert not dfa.is_accepting("abc-de")

def test_wildcard():
    nfa = FiniteStateMachine.from_regex("a.b")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("acb")
    assert dfa.is_accepting("a#b")
    assert not dfa.is_accepting("acdb")
    assert not dfa.is_accepting("abbb")
    assert not dfa.is_accepting("ab")
    assert not dfa.is_accepting("a\nb")
    assert dfa.is_accepting("a\tb")

def test_wildcard_and_plus():
    nfa = FiniteStateMachine.from_regex("a.+b")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("acb")
    assert dfa.is_accepting("axyzb")
    assert not dfa.is_accepting("ab")
    # dot does not match newline by default
    assert not dfa.is_accepting("a\nb")

def test_optional():
    # "colou?r" should accept "color" or "colour"
    nfa = FiniteStateMachine.from_regex("colou?r")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("color")
    assert dfa.is_accepting("colour")
    assert not dfa.is_accepting("colr")
    assert not dfa.is_accepting("colourr")

def test_nested_groups_and_alternation():
    # Pattern: ((ab|cd)+e?)f
    nfa = FiniteStateMachine.from_regex("((ab|cd)+e?)f")
    dfa = nfa.to_dfa()
    # Valid strings:
    assert dfa.is_accepting("abf")         # “ab” once, no “e”
    assert dfa.is_accepting("abe f".replace(" ", ""))  # “ab” + “e” + “f”
    assert dfa.is_accepting("cdef")        # “cd” once, no “e”
    assert dfa.is_accepting("abcdabef")    # “ab cd ab” + “e” + “f”
    assert dfa.is_accepting("cdabe f".replace(" ", ""))# “cd” + “ab” + “e” + “f”
    # Invalid strings:
    assert not dfa.is_accepting("cdabcf")
    assert not dfa.is_accepting("f")         # neither “ab” nor “cd” appears
    assert not dfa.is_accepting("ef")         # neither “ab” nor “cd” appears
    assert not dfa.is_accepting("abe")       # missing trailing “f”
    assert not dfa.is_accepting("abcef")     # “ce” is not “cd” or “ab”
    assert not dfa.is_accepting("abcd")      # missing final “f”

def test_character_class_ranges_and_unions():
    # Pattern: [A-CX-Z0-3]{2,4}  (choose 2–4 chars from A–C or X–Z or 0–3)
    nfa = FiniteStateMachine.from_regex("[A-CX-Z0-3]{2,4}")
    dfa = nfa.to_dfa()
    # Valid: length between 2 and 4, each character in one of those subranges
    assert dfa.is_accepting("AB")            # both in A–C
    assert dfa.is_accepting("CX")            # “C” from A–C, “X” from X–Z
    assert dfa.is_accepting("0Z03")          # “0” from 0–3, “Z” from X–Z, “0” from 0–3, “3” from 0–3
    assert dfa.is_accepting("XYZ")           # length 3, all X–Z
    # Too short / too long / invalid chars
    assert not dfa.is_accepting("A")         # length 1
    assert not dfa.is_accepting("ABCD5")     # length 5
    assert not dfa.is_accepting("AZ*")       # “*” is invalid
    assert not dfa.is_accepting("BD")        # “B” is valid but “D” is not in any of the ranges

def test_mixed_classes_and_word_boundary_like():
    # Pattern: \w+\s\d{1,2}   (one or more word chars, then a whitespace, then 1–2 digits)
    nfa = FiniteStateMachine.from_regex(r"\w+\s\d{1,2}")
    dfa = nfa.to_dfa()
    # \w is [A-Za-z0-9_], \s is whitespace
    assert dfa.is_accepting("abc_123 5")
    assert dfa.is_accepting("X1 99")
    assert dfa.is_accepting("_Z9\t0")        # underscore, letter, digit, then tab, then one digit
    # Missing pieces or extra characters
    assert not dfa.is_accepting(" abc 5")     # leading space: begins with \w+ fails
    assert not dfa.is_accepting("abc   5")    # too many spaces: \s matches exactly one
    assert not dfa.is_accepting("abc 123")    # “123” is three digits, wants 1–2
    assert not dfa.is_accepting("ab$ 5")      # “$” is not \w
    assert not dfa.is_accepting("abc  ")      # missing the digit(s)

def test_combined_star_plus_and_optional():
    # Pattern: a(bc)*d+e?f
    nfa = FiniteStateMachine.from_regex("a(bc)*d+e?f")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("adf")           # zero “bc”, one “d”, no “e”
    assert dfa.is_accepting("abcddef")       # “a” + “bc” + “d” + “d” + “e” + “f”
    assert dfa.is_accepting("abcbcddef")     # two “bc”, two “d”, “e”, “f”
    assert dfa.is_accepting("abcbcddef")     # same string again
    assert dfa.is_accepting("abcbcd f".replace(" ", ""))  # no “e”
    assert dfa.is_accepting("abcbcddef")     # repeated test
    # Negative cases
    assert not dfa.is_accepting("af")        # missing required “d+”
    assert not dfa.is_accepting("ad")        # missing final “f”
    assert not dfa.is_accepting("abcdde")    # missing final “f”
    assert not dfa.is_accepting("abcddefg")  # extra “g”

def test_long_chain_of_nested_alternation():
    # Pattern: (a|(b|(c|(d|e)))){1,3}
    # effectively any of {a, b, c, d, e} repeated 1–3 times
    nfa = FiniteStateMachine.from_regex("(a|(b|(c|(d|e)))){1,3}")
    dfa = nfa.to_dfa()
    # All length‐1,2,3 strings over {a,b,c,d,e}:
    for length in (1, 2, 3):
        for letters in ["a", "b", "c", "d", "e"]:
            # e.g. "a", "aa", "aaa", "ab", "abc", etc.
            import itertools
            for tup in itertools.product("abcde", repeat=length):
                s = "".join(tup)
                assert dfa.is_accepting(s), f"should accept {s}"
    # Too short (empty) or too long or invalid chars
    assert not dfa.is_accepting("")
    assert not dfa.is_accepting("abcd")      # length 4
    assert not dfa.is_accepting("z")         # “z” not in group
    assert not dfa.is_accepting("ab1")       # “1” not in group

def test_complex_email_like():
    # A tiny “email‐like” pattern:  \w+@[a-z]+\.com
    nfa = FiniteStateMachine.from_regex(r"\w+@[a-z]+\.com")
    dfa = nfa.to_dfa()

    assert dfa.is_accepting("alice@xyz.com")
    assert dfa.is_accepting("bob_123@abc.com")
    assert dfa.is_accepting("X9@z.com")
    # Negative:
    assert not dfa.is_accepting("noatsymbol.com")
    assert not dfa.is_accepting("user@domain.net")   # .net not .com
    assert not dfa.is_accepting("user@domaincom")    # missing “.”
    assert not dfa.is_accepting("@abc.com")          # missing local‐part
    assert not dfa.is_accepting("alice@XYZ.com")     # uppercase in domain, [a-z]+ only

def test_literals_and_digits():
    nfa = FiniteStateMachine.from_regex("[a-h]{2}\d")
    dfa = nfa.to_dfa()

    assert dfa.is_accepting("bg9")
    assert dfa.is_accepting("hh0")

    # Negative: 
    assert not dfa.is_accepting("e7")   # only 1 literal + digit
    assert not dfa.is_accepting("dc")  # missing digit
    assert not dfa.is_accepting("ia7")       # “a” not in Greek‐lower range

def test_unicode_literals_and_digits():
    # Although Python string literal handles \u before parsing, just test a simple Unicode run:
    # Pattern: [α-ω]{2}\d  (two lowercase Greek letters, then one digit)
    # Here: α (U+03B1) … ω (U+03C9)
    greek_lower = "\u03B1-\u03C9"  # becomes the two literal chars “α-ω” but in a class they mean a range
    pattern = f"[{greek_lower}]{{2}}\\d"
    nfa = FiniteStateMachine.from_regex(pattern)
    dfa = nfa.to_dfa()

    # Pick specific codepoints: “β” (U+03B2), “μ” (U+03BC), then digit “7”
    test_string = "\u03B2\u03BC7"   # “βμ7”
    assert dfa.is_accepting(test_string)

    # Negative: wrong number of Greek letters or missing digit
    assert not dfa.is_accepting("\u03B27")   # only 1 Greek + digit
    assert not dfa.is_accepting("\u03B2\u03BC")  # missing digit
    assert not dfa.is_accepting("ab7")       # “a” not in Greek‐lower range

def test_empty_pattern_and_only_epsilon():
    # An empty pattern (""), by many conventions, matches only the empty string
    nfa = FiniteStateMachine.from_regex("")
    print(nfa.transitions)
    dfa = nfa.to_dfa()
    print(dfa.transitions)
    assert dfa.is_accepting("")             # empty string
    assert not dfa.is_accepting("a")
    assert not dfa.is_accepting(" ")

    # A pattern that is literally epsilon (via grouping an empty): "(?:)"
    nfa2 = FiniteStateMachine.from_regex("()")
    dfa2 = nfa2.to_dfa()
    assert dfa2.is_accepting("")
    assert not dfa2.is_accepting("anything")

def test_combined_question_and_union():
    # Pattern: (red|green|blue)?-?\d+  – optional color – optional hyphen – one or more digits
    pat = "(red|green|blue)?-?\\d+"
    nfa = FiniteStateMachine.from_regex(pat)
    dfa = nfa.to_dfa()
    # Should accept:
    assert dfa.is_accepting("123")
    assert dfa.is_accepting("-7")
    assert dfa.is_accepting("red-456")
    assert dfa.is_accepting("green789")
    assert dfa.is_accepting("blue-0")
    # Should reject:
    assert not dfa.is_accepting("")           # needs at least \d+
    assert not dfa.is_accepting("-")          # no digit
    assert not dfa.is_accepting("yellow-5")   # “yellow” not in the union
    assert not dfa.is_accepting("redgreen-2") # “redgreen” not exactly one of the three

def test_complex_nesting_and_quantifier_mix():
    # Pattern: ((a|b){2,3}c?)+d{1,2}
    nfa = FiniteStateMachine.from_regex("((a|b){2,3}c?)+d{1,2}")
    dfa = nfa.to_dfa()
    # Should accept:
    assert dfa.is_accepting("aacdd")     # (aa + no c) once, + dd
    assert dfa.is_accepting("bbc d".replace(" ", "")) # (bb + no c) once + d
    assert dfa.is_accepting("abc d".replace(" ", "")) # (ab + c) once + d
    assert dfa.is_accepting("abca abd d".replace(" ", ""))  
    assert dfa.is_accepting("abcd")
    assert dfa.is_accepting("aaaa d".replace(" ", ""))
    assert dfa.is_accepting("ababcd")
    # Negative cases:
    assert not dfa.is_accepting("a")      # too short   
    assert not dfa.is_accepting("abc")    # no d at end
    assert not dfa.is_accepting("abcbdd") # (ab c) once, okay, dd is okay, but “b” leftover in the middle? Actually “abcbdd” = “abc” + “b d d”; “b” is extra.

def test_lower_and_upper_bound_edges():
    # Pattern: x{0,0} should match only "", and y{2,2} should match exactly “yy”
    nfa1 = FiniteStateMachine.from_regex("x{0,0}")
    dfa1 = nfa1.to_dfa()
    assert dfa1.is_accepting("")
    assert not dfa1.is_accepting("x")

    nfa2 = FiniteStateMachine.from_regex("y{2,2}")
    dfa2 = nfa2.to_dfa()
    assert dfa2.is_accepting("yy")
    assert not dfa2.is_accepting("y")
    assert not dfa2.is_accepting("yyy")

def test_zero_or_more_of_empty_group():
    # Pattern: (ab?c)* 
    nfa = FiniteStateMachine.from_regex("(ab?c)*")
    dfa = nfa.to_dfa()
    assert dfa.is_accepting("")        # zero repetitions
    assert dfa.is_accepting("ac")      # (a c)
    assert dfa.is_accepting("abc")     # (a b c)
    assert dfa.is_accepting("acabc")   # (ac)(abc)
    assert dfa.is_accepting("abcac")   # (abc)(ac)
    assert dfa.is_accepting("acac")    # (ac)(ac)
    assert not dfa.is_accepting("a")   # incomplete
    assert not dfa.is_accepting("ab")  # incomplete
    assert not dfa.is_accepting("cab") # wrong order


def test_whitespace_and_alphanumeric():
    # Pattern: \s([A-Za-z0-9]{1,5}([.!?,]\s|\s))+
    nfa = FiniteStateMachine.from_regex(r"\s([A-Za-z0-9]{1,5}([.!?,]\s|\s))+")
    dfa = nfa.to_dfa()

    # Valid strings (should accept)
    assert dfa.is_accepting(" a. ")          # starts with whitespace, valid sequence
    assert dfa.is_accepting(" b! c? ")        # multiple valid sequences with punctuation
    assert dfa.is_accepting(" 1234, ")        # valid numeric sequence followed by punctuation
    assert dfa.is_accepting(" test ")         # valid alphanumeric followed by space
    assert dfa.is_accepting(" hello! world? ") # multiple valid sequences with punctuation

    # Invalid strings (should not accept)
    assert not dfa.is_accepting("a.")         # does not start with whitespace
    assert not dfa.is_accepting("  ")         # only whitespace, no valid sequence
    assert not dfa.is_accepting(" 123456! ")   # valid but too long (6 characters)
    assert not dfa.is_accepting(" test. test") # valid but does not end with space or punctuation
    assert not dfa.is_accepting("!test ")      # starts with punctuation, invalid
    assert not dfa.is_accepting("")             # empty string
    assert not dfa.is_accepting("abcde")       # no leading whitespace and no valid sequence