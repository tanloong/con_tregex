#!/usr/bin/env python3
# -*- coding=utf-8 -*-

# https://github.com/stanfordnlp/CoreNLP/blob/efc66a9cf49fecba219dfaa4025315ad966285cc/test/src/edu/stanford/nlp/trees/tregex/TregexTest.java

from .base_tmpl import BaseTmpl
from .base_tmpl import tree as tree_string

from con_tregex.tregex import TregexPattern

class TestTree(BaseTmpl):
    def setUp(self):
        self.tree_string = tree_string
        return super().setUp()

    def test_JoãoSilva(self):
        tregex1 = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, __=m <- (__ <, CONJ <- __=z))))")
        tregex2 = TregexPattern("PNT=p >>- (/(.+)/#1%var=l >, (__=t <- (__=r <, /(.+)/#1%var=m <- (__ <, CONJ <- /(.+)/#1%var=z))))")
        tregex3 = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, ~l <- (__ <, CONJ <- ~l))))")
        tree_string = "(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))"

        self.assertTrue(tregex1.findall(tree_string))
        # self.assertTrue(tregex2.findall(tree_string))
        # self.assertTrue(tregex3.findall(tree_string))
