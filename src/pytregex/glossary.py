#!/usr/bin/env python3

# https://nlp.stanford.edu/nlp/javadoc/javanlp-3.5.0/edu/stanford/nlp/trees/tregex/TregexPattern.html

import warnings


def explain(term):
    if term in GLOSSARY:
        return GLOSSARY[term]
    else:
        warnings.warn(f"Term '{term}' not found in glossary.", stacklevel=1)


GLOSSARY = {
    "<<": "A << B\nA dominates B",
    ">>": "A >> B\nA is dominated by B",
    "<": "A < B\nA immediately dominates B",
    ">": "A > B\nA is immediately dominated by B",
    "$": "A $ B\nA is a sister of B (and not equal to B)",
    "..": "A .. B\nA precedes B",
    ".": "A . B\nA immediately precedes B",
    ",,": "A ,, B\nA follows B",
    ",": "A , B\nA immediately follows B",
    "<<,": "A <<, B\nB is a leftmost descendant of A",
    "<<-": "A <<- B\nB is a rightmost descendant of A",
    ">>,": "A >>, B\nA is a leftmost descendant of B",
    ">>-": "A >>- B\nA is a rightmost descendant of B",
    "<,": "A <, B\nB is the first child of A",
    ">,": "A >, B\nA is the first child of B",
    "<-": "A <- B\nB is the last child of A",
    ">-": "A >- B\nA is the last child of B",
    "<`": "A <` B\nB is the last child of A",
    ">`": "A >` B\nA is the last child of B",
    "<i": "A <i B\nB is the ith child of A (i > 0)",
    ">i": "A >i B\nA is the ith child of B (i > 0)",
    "<-i": "A <-i B\nB is the ith-to-last child of A (i > 0)",
    ">-i": "A >-i B\nA is the ith-to-last child of B (i > 0)",
    "<:": "A <: B\nB is the only child of A",
    ">:": "A >: B\nA is the only child of B",
    "<<:": "A <<: B\nA dominates B via an unbroken chain (length > 0) of unary local trees.",
    ">>:": "A >>: B\nA is dominated by B via an unbroken chain (length > 0) of unary local trees.",
    "$++": "A $++ B\nA is a left sister of B (same as $.. for context-free trees)",
    "$--": "A $-- B\nA is a right sister of B (same as $,, for context-free trees)",
    "$+": "A $+ B\nA is the immediate left sister of B (same as $. for context-free trees)",
    "$-": "A $- B\nA is the immediate right sister of B (same as $, for context-free trees)",
    "$..": "A $.. B\nA is a sister of B and precedes B",
    "$,,": "A $,, B\nA is a sister of B and follows B",
    "$.": "A $. B\nA is a sister of B and immediately precedes B",
    "$,": "A $, B\nA is a sister of B and immediately follows B",
    "<+(C)": "A <+(C) B\nA dominates B via an unbroken chain of (zero or more) nodes matching description C",
    ">+(C)": "A >+(C) B\nA is dominated by B via an unbroken chain of (zero or more) nodes matching description C",
    ".+(C)": "A .+(C) B\nA precedes B via an unbroken chain of (zero or more) nodes matching description C",
    ",+(C)": "A ,+(C) B\nA follows B via an unbroken chain of (zero or more) nodes matching description C",
    "<<#": "A <<# B\nB is a head of phrase A",
    ">>#": "A >># B\nA is a head of phrase B",
    "<#": "A <# B\nB is the immediate head of phrase A",
    ">#": "A ># B\nA is the immediate head of phrase B",
    "==": "A == B\nA and B are the same node",
    "<=": "A <= B\nA and B are the same node or A is the parent of B",
    ":": "A : B\n[this is a pattern-segmenting operator that places no constraints on the relationship between A and B]",
    "<...": "A <... { B ; C ; ... }	-- A has exactly B, C, etc as its subtree, with no other children.",
}
