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
