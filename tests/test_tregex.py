#!/usr/bin/env python3
# -*- coding=utf-8 -*-

# translated from https://github.com/stanfordnlp/CoreNLP/blob/efc66a9cf49fecba219dfaa4025315ad966285cc/test/src/edu/stanford/nlp/trees/tregex/TregexTest.java
# last modified at Apr 2, 2022 (https://github.com/stanfordnlp/CoreNLP/commits/efc66a9cf49fecba219dfaa4025315ad966285cc/test/src/edu/stanford/nlp/trees/tregex/TregexTest.java)

from typing import Union

from pytregex.tregex import TregexPattern
from pytregex.tree import Tree

from .base_tmpl import BaseTmpl
from .base_tmpl import tree as tree_string


class TestTregex(BaseTmpl):
    def setUp(self):
        self.tree_string = tree_string
        return super().setUp()

    def test_JoãoSilva(self):
        tregex1 = TregexPattern(
            "PNT=p >>- (__=l >, (__=t <- (__=r <, __=m <- (__ <, CONJ <- __=z))))"
        )
        tregex2 = TregexPattern(
            "PNT=p >>- (/(.+)/#1%var=l >, (__=t <- (__=r <, /(.+)/#1%var=m <- (__ <, CONJ <-"
            " /(.+)/#1%var=z))))"
        )
        tregex3 = TregexPattern(
            "PNT=p >>- (__=l >, (__=t <- (__=r <, ~l <- (__ <, CONJ <- ~l))))"
        )
        tree_string = (
            "(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))"
        )

        self.assertTrue(tregex1.findall(tree_string))
        # TODO
        # self.assertTrue(tregex2.findall(tree_string))
        # self.assertTrue(tregex3.findall(tree_string))

    def test_no_results(self):
        pMWE = TregexPattern("/^MW/")
        self.assertFalse(pMWE.findall("(Foo)"))

    def test_one_result(self):
        pMWE = TregexPattern("/^MW/")
        matches = pMWE.findall("(ROOT (MWE (N 1) (N 2) (N 3)))")
        self.assertEqual(1, len(matches))
        self.assertEqual("(MWE (N 1) (N 2) (N 3))", matches[0].to_string())

    def test_two_results(self):
        pMWE = TregexPattern("/^MW/")
        matches = pMWE.findall("(ROOT (MWE (N 1) (N 2) (N 3)) (MWV (A B)))")
        self.assertEqual(2, len(matches))

        self.assertEqual("(MWE (N 1) (N 2) (N 3))", matches[0].to_string())
        self.assertEqual("(MWV (A B))", matches[1].to_string())

    def test_reuse(self):
        """
        a tregex pattern should be able to go more than once. just like me.
        """
        pMWE = TregexPattern("/^MW/")

        matches = pMWE.findall("(ROOT (MWE (N 1) (N 2) (N 3)) (MWV (A B)))")
        self.assertEqual(2, len(matches))

        matches = pMWE.findall("(ROOT (MWE (N 1) (N 2) (N 3)))")
        self.assertEqual(1, len(matches))

        matches = pMWE.findall("(Foo)")
        self.assertEqual(0, len(matches))

    def test_ith_child(self):
        # A is the ith child of B
        self.run_test(
            "/.*/ >1 root",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
        )
        self.run_test("/.*/ >1 a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test("/.*/ >2 a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")
        self.run_test("/.*/ >1 foo", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(1)")
        self.run_test("/.*/ >2 foo", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(2)")
        self.run_test("/.*/ >1 bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(3)")
        self.run_test("/.*/ >2 bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(4)")

        # A is the -ith child of B
        self.run_test(
            "/.*/ >-1 root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(b (baz 5))"
        )
        self.run_test("/.*/ >-1 a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")
        self.run_test("/.*/ >-2 a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test("/.*/ >-1 foo", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(2)")
        self.run_test("/.*/ >-2 bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(3)")
        self.run_test("/.*/ >-1 bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(4)")
        self.run_test("/.*/ >-1 b", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(baz 5)")
        self.run_test("/.*/ >-2 b", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")

        # B is the ith child of A
        self.run_test("/.*/ <1 root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test(
            "/.*/ <1 a",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
        )
        self.run_test("/.*/ <1 /1/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test("/.*/ <1 /2/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <1 bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test(
            "/.*/ <2 bar",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
        )
        self.run_test("/.*/ <3 bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <1 /3/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")
        self.run_test("/.*/ <1 /4/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <2 /4/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")

        # B is the -ith child of A
        self.run_test("/.*/ <-1 root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <-1 a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test(
            "/.*/ <-2 a",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
        )
        self.run_test("/.*/ <-1 /1/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <-2 /1/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test("/.*/ <-1 /2/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test("/.*/ <-2 /2/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test(
            "/.*/ <-1 bar",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
        )
        self.run_test("/.*/ <-1 /3/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <-2 /3/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")
        self.run_test("/.*/ <-1 /4/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")

    def test_test(self):
        """
        reruns one of the simpler tests using the test class to make sure
        the test class works
        """
        pattern = TregexPattern("/^MW/")
        self.run_test(
            pattern,
            "(ROOT (MWE (N 1) (N 2) (N 3)) (MWV (A B)))",
            "(MWE (N 1) (N 2) (N 3))",
            "(MWV (A B))",
        )

    def test_word_disjunction(self):
        pattern = TregexPattern("a|b|c << bar")
        self.run_test(pattern, "(a (bar 1))", "(a (bar 1))")
        self.run_test(pattern, "(b (bar 1))", "(b (bar 1))")
        self.run_test(pattern, "(c (bar 1))", "(c (bar 1))")
        self.run_test(pattern, "(d (bar 1))")
        self.run_test(pattern, "(e (bar 1))")
        self.run_test(pattern, "(f (bar 1))")
        self.run_test(pattern, "(g (bar 1))")

        pattern = TregexPattern("a|b|c|d|e|f << bar")
        self.run_test(pattern, "(a (bar 1))", "(a (bar 1))")
        self.run_test(pattern, "(b (bar 1))", "(b (bar 1))")
        self.run_test(pattern, "(c (bar 1))", "(c (bar 1))")
        self.run_test(pattern, "(d (bar 1))", "(d (bar 1))")
        self.run_test(pattern, "(e (bar 1))", "(e (bar 1))")
        self.run_test(pattern, "(f (bar 1))", "(f (bar 1))")
        self.run_test(pattern, "(g (bar 1))")

    def test_dominates(self):
        dominates_pattern = TregexPattern("foo << bar")
        self.run_test(dominates_pattern, "(foo (bar 1))", "(foo (bar 1))")
        self.run_test(dominates_pattern, "(foo (a (bar 1)))", "(foo (a (bar 1)))")
        self.run_test(dominates_pattern, "(foo (a (b (bar 1))))", "(foo (a (b (bar 1))))")
        self.run_test(dominates_pattern, "(foo (a (b 1) (bar 2)))", "(foo (a (b 1) (bar 2)))")
        self.run_test(
            dominates_pattern, "(foo (a (b 1) (c 2) (bar 3)))", "(foo (a (b 1) (c 2) (bar 3)))"
        )
        self.run_test(dominates_pattern, "(foo (baz 1))")
        self.run_test(dominates_pattern, "(a (foo (bar 1)))", "(foo (bar 1))")
        self.run_test(dominates_pattern, "(a (foo (baz (bar 1))))", "(foo (baz (bar 1)))")
        self.run_test(
            dominates_pattern,
            "(a (foo (bar 1)) (foo (bar 2)))",
            "(foo (bar 1))",
            "(foo (bar 2))",
        )

        dominated_pattern = TregexPattern("foo >> bar")
        self.run_test(dominated_pattern, "(foo (bar 1))")
        self.run_test(dominated_pattern, "(foo (a (bar 1)))")
        self.run_test(dominated_pattern, "(foo (a (b (bar 1))))")
        self.run_test(dominated_pattern, "(foo (a (b 1) (bar 2)))")
        self.run_test(dominated_pattern, "(foo (a (b 1) (c 2) (bar 3)))")
        self.run_test(dominated_pattern, "(bar (foo 1))", "(foo 1)")
        self.run_test(dominated_pattern, "(bar (a (foo 1)))", "(foo 1)")
        self.run_test(dominated_pattern, "(bar (a (foo (b 1))))", "(foo (b 1))")
        self.run_test(dominated_pattern, "(bar (a (foo 1) (foo 2)))", "(foo 1)", "(foo 2)")
        self.run_test(dominated_pattern, "(bar (foo (foo 1)))", "(foo (foo 1))", "(foo 1)")
        self.run_test(dominated_pattern, "(a (bar (foo 1)))", "(foo 1)")

    def test_immediately_dominates(self):
        dominates_pattern = TregexPattern("foo < bar")
        self.run_test(dominates_pattern, "(foo (bar 1))", "(foo (bar 1))")
        self.run_test(dominates_pattern, "(foo (a (bar 1)))")
        self.run_test(dominates_pattern, "(a (foo (bar 1)))", "(foo (bar 1))")
        self.run_test(dominates_pattern, "(a (foo (baz 1) (bar 2)))", "(foo (baz 1) (bar 2))")
        self.run_test(
            dominates_pattern,
            "(a (foo (bar 1)) (foo (bar 2)))",
            "(foo (bar 1))",
            "(foo (bar 2))",
        )

        dominated_pattern = TregexPattern("foo > bar")
        self.run_test(dominated_pattern, "(foo (bar 1))")
        self.run_test(dominated_pattern, "(foo (a (bar 1)))")
        self.run_test(dominated_pattern, "(foo (a (b (bar 1))))")
        self.run_test(dominated_pattern, "(foo (a (b 1) (bar 2)))")
        self.run_test(dominated_pattern, "(foo (a (b 1) (c 2) (bar 3)))")
        self.run_test(dominated_pattern, "(bar (foo 1))", "(foo 1)")
        self.run_test(dominated_pattern, "(bar (a (foo 1)))")
        self.run_test(dominated_pattern, "(bar (foo 1) (foo 2))", "(foo 1)", "(foo 2)")
        self.run_test(dominated_pattern, "(bar (foo (foo 1)))", "(foo (foo 1))")
        self.run_test(dominated_pattern, "(a (bar (foo 1)))", "(foo 1)")

    def test_sister(self):
        pattern = TregexPattern("/.*/ $ foo")
        self.run_test(pattern, "(a (foo 1) (bar 2))", "(bar 2)")
        self.run_test(pattern, "(a (bar 1) (foo 2))", "(bar 1)")
        self.run_test(pattern, "(a (foo 1) (bar 2) (baz 3))", "(bar 2)", "(baz 3)")
        self.run_test(pattern, "(a (foo (bar 2)) (baz 3))", "(baz 3)")
        self.run_test(pattern, "(a (foo (bar 2)) (baz (bif 3)))", "(baz (bif 3))")
        self.run_test(pattern, "(a (foo (bar 2)))")
        self.run_test(pattern, "(a (foo 1))")

        pattern = TregexPattern("bar|baz $ foo")
        self.run_test(pattern, "(a (foo 1) (bar 2))", "(bar 2)")
        self.run_test(pattern, "(a (bar 1) (foo 2))", "(bar 1)")
        self.run_test(pattern, "(a (foo 1) (bar 2) (baz 3))", "(bar 2)", "(baz 3)")
        self.run_test(pattern, "(a (foo (bar 2)) (baz 3))", "(baz 3)")
        self.run_test(pattern, "(a (foo (bar 2)) (baz (bif 3)))", "(baz (bif 3))")
        self.run_test(pattern, "(a (foo (bar 2)))")
        self.run_test(pattern, "(a (foo 1))")

        pattern = TregexPattern("/.*/ $ foo")
        self.run_test(pattern, "(a (foo 1) (foo 2))", "(foo 1)", "(foo 2)")
        self.run_test(pattern, "(a (foo 1))")

        pattern = TregexPattern("foo $ foo")
        self.run_test(pattern, "(a (foo 1) (foo 2))", "(foo 1)", "(foo 2)")
        self.run_test(pattern, "(a (foo 1))")

        # TODO
        # pattern = TregexPattern("foo $ foo=a")
        # tree = Tree.from_string("(a (foo 1) (foo 2) (foo 3))")
        # matcher = pattern.findall(tree)
        # self.assertTrue(matcher.find())
        # self.assertEquals("(foo 1)", matcher.getMatch().toString())
        # self.assertEquals("(foo 2)", matcher.getNode("a").toString())
        # self.assertTrue(matcher.find())
        # self.assertEquals("(foo 1)", matcher.getMatch().toString())
        # self.assertEquals("(foo 3)", matcher.getNode("a").toString())
        # self.assertTrue(matcher.find())
        # self.assertEquals("(foo 2)", matcher.getMatch().toString())
        # self.assertEquals("(foo 1)", matcher.getNode("a").toString())
        # self.assertTrue(matcher.find())
        # self.assertEquals("(foo 2)", matcher.getMatch().toString())
        # self.assertEquals("(foo 3)", matcher.getNode("a").toString())
        # self.assertTrue(matcher.find())
        # self.assertEquals("(foo 3)", matcher.getMatch().toString())
        # self.assertEquals("(foo 1)", matcher.getNode("a").toString())
        # self.assertTrue(matcher.find())
        # self.assertEquals("(foo 3)", matcher.getMatch().toString())
        # self.assertEquals("(foo 2)", matcher.getNode("a").toString())
        # self.assertFalse(matcher.find())
        # self.run_test("foo $ foo", "(a (foo 1))")

    def test_precedes_follows(self):
        # precedes
        pattern = TregexPattern("/.*/ .. foo")

        self.run_test(pattern, "(a (foo 1) (bar 2))")
        self.run_test(pattern, "(a (bar 1) (foo 2))", "(bar 1)", "(1)")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo 3))", "(bar 1)", "(1)", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (foo 1) (baz 2) (bar 3))")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2))")
        self.run_test(pattern, "(a (bar 1) (baz (foo 2)))", "(bar 1)", "(1)")
        self.run_test(
            pattern, "(a (bar 1) (baz 2) (bif (foo 3)))", "(bar 1)", "(1)", "(baz 2)", "(2)"
        )
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2) (bif 3))")
        self.run_test(
            pattern, "(a (bar 1) (baz 2) (foo (bif 3)))", "(bar 1)", "(1)", "(baz 2)", "(2)"
        )
        self.run_test(pattern, "(a (bar 1) (foo (bif 2)) (baz 3))", "(bar 1)", "(1)")

        # follows
        pattern = TregexPattern("/.*/ ,, foo")

        self.run_test(pattern, "(a (foo 1) (bar 2))", "(bar 2)", "(2)")
        self.run_test(pattern, "(a (bar 1) (foo 2))")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo 3))")
        self.run_test(pattern, "(a (foo 1) (baz 2) (bar 3))", "(baz 2)", "(2)", "(bar 3)", "(3)")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2))", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (bar 1) (baz (foo 2)))")
        self.run_test(pattern, "(a (bar 1) (baz 2) (bif (foo 3)))")
        self.run_test(
            pattern, "(a (bar (foo 1)) (baz 2) (bif 3))", "(baz 2)", "(2)", "(bif 3)", "(3)"
        )
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo (bif 3)))")
        self.run_test(
            pattern, "(a (foo (bif 1)) (bar 2) (baz 3))", "(bar 2)", "(2)", "(baz 3)", "(3)"
        )
        self.run_test(pattern, "(a (bar 1) (foo (bif 2)) (baz 3))", "(baz 3)", "(3)")

    def test_immediate_precedes_follows(self):
        # immediate precedes
        pattern = TregexPattern("/.*/ . foo")

        self.run_test(pattern, "(a (foo 1) (bar 2))")
        self.run_test(pattern, "(a (bar 1) (foo 2))", "(bar 1)", "(1)")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo 3))", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (foo 1) (baz 2) (bar 3))")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2))")
        self.run_test(pattern, "(a (bar 1) (baz (foo 2)))", "(bar 1)", "(1)")
        self.run_test(pattern, "(a (bar 1) (baz 2) (bif (foo 3)))", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2) (bif 3))")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo (bif 3)))", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (bar 1) (foo (bif 2)) (baz 3))", "(bar 1)", "(1)")
        self.run_test(
            pattern,
            "(a (bar 1) (foo 2) (baz 3) (foo 4) (bif 5))",
            "(bar 1)",
            "(1)",
            "(baz 3)",
            "(3)",
        )
        self.run_test(
            pattern, "(a (bar 1) (foo 2) (foo 3) (baz 4))", "(bar 1)", "(1)", "(foo 2)", "(2)"
        )
        self.run_test(pattern, "(a (b (c 1) (d 2)) (foo))", "(b (c 1) (d 2))", "(d 2)", "(2)")
        self.run_test(
            pattern, "(a (b (c 1) (d 2)) (bar (foo 3)))", "(b (c 1) (d 2))", "(d 2)", "(2)"
        )
        self.run_test(pattern, "(a (b (c 1) (d 2)) (bar (baz 3) (foo 4)))", "(baz 3)", "(3)")
        self.run_test(pattern, "(a (b (c 1) (d 2)) (bar (baz 2 3) (foo 4)))", "(baz 2 3)", "(3)")

        # immediate follows
        pattern = TregexPattern("/.*/ , foo")

        self.run_test(pattern, "(a (foo 1) (bar 2))", "(bar 2)", "(2)")
        self.run_test(pattern, "(a (bar 1) (foo 2))")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo 3))")
        self.run_test(pattern, "(a (foo 1) (baz 2) (bar 3))", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2))", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (bar 1) (baz (foo 2)))")
        self.run_test(pattern, "(a (bar 1) (baz 2) (bif (foo 3)))")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2) (bif 3))", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo (bif 3)))")
        self.run_test(pattern, "(a (foo (bif 1)) (bar 2) (baz 3))", "(bar 2)", "(2)")
        self.run_test(pattern, "(a (bar 1) (foo (bif 2)) (baz 3))", "(baz 3)", "(3)")
        self.run_test(
            pattern,
            "(a (bar 1) (foo 2) (baz 3) (foo 4) (bif 5))",
            "(baz 3)",
            "(3)",
            "(bif 5)",
            "(5)",
        )
        self.run_test(
            pattern, "(a (bar 1) (foo 2) (foo 3) (baz 4))", "(foo 3)", "(3)", "(baz 4)", "(4)"
        )
        self.run_test(pattern, "(a (foo) (b (c 1) (d 2)))", "(b (c 1) (d 2))", "(c 1)", "(1)")
        self.run_test(
            pattern, "(a (bar (foo 3)) (b (c 1) (d 2)))", "(b (c 1) (d 2))", "(c 1)", "(1)"
        )
        self.run_test(
            pattern,
            "(a (bar (baz 3) (foo 4)) (b (c 1) (d 2)))",
            "(b (c 1) (d 2))",
            "(c 1)",
            "(1)",
        )
        self.run_test(pattern, "(a (bar (foo 4) (baz 3)) (b (c 1) (d 2)))", "(baz 3)", "(3)")

    def test_left_right_most_descendant(self):
        # B leftmost descendant of A
        self.run_test(
            "/.*/ <<, /1/",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
            "(foo 1 2)",
        )
        self.run_test("/.*/ <<, /2/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test(
            "/.*/ <<, foo",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
        )
        self.run_test(
            "/.*/ <<, baz", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(b (baz 5))"
        )

        # B rightmost descendant of A
        self.run_test("/.*/ <<- /1/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <<- /2/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test(
            "/.*/ <<- /4/",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
            "(bar 3 4)",
        )

        # A leftmost descendant of B
        self.run_test(
            "/.*/ >>, root",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
            "(foo 1 2)",
            "(1)",
        )
        self.run_test(
            "/.*/ >>, a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)", "(1)"
        )
        self.run_test("/.*/ >>, bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(3)")

        # A rightmost descendant of B
        self.run_test(
            "/.*/ >>- root",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(b (baz 5))",
            "(baz 5)",
            "(5)",
        )
        self.run_test(
            "/.*/ >>- a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)", "(4)"
        )
        self.run_test("/.*/ >>- /1/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")

    def test_first_last_child(self):
        # A is the first child of B
        self.run_test(
            "/.*/ >, root",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
        )
        self.run_test("/.*/ >, a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test("/.*/ >, foo", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(1)")
        self.run_test("/.*/ >, bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(3)")

        # A is the last child of B
        self.run_test(
            "/.*/ >- root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(b (baz 5))"
        )
        self.run_test("/.*/ >- a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")
        self.run_test("/.*/ >- foo", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(2)")
        self.run_test("/.*/ >- bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(4)")
        self.run_test("/.*/ >- b", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(baz 5)")

        # B is the first child of A
        self.run_test("/.*/ <, root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test(
            "/.*/ <, a",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
        )
        self.run_test("/.*/ <, /1/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test("/.*/ <, /2/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <, bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <, /3/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")
        self.run_test("/.*/ <, /4/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")

        # B is the last child of A
        self.run_test("/.*/ <- root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <- a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <- /1/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <- /2/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)")
        self.run_test(
            "/.*/ <- bar",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(a (foo 1 2) (bar 3 4))",
        )
        self.run_test("/.*/ <- /3/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))")
        self.run_test("/.*/ <- /4/", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)")

    def test_only_child(self):
        self.run_test("foo <: bar", "(foo (bar 1))", "(foo (bar 1))")
        self.run_test("foo <: bar", "(foo (bar 1) (bar 2))")
        self.run_test("foo <: bar", "(foo)")
        self.run_test("foo <: bar", "(foo (baz (bar)))")
        self.run_test("foo <: bar", "(foo 1)")

        self.run_test("bar >: foo", "(foo (bar 1))", "(bar 1)")
        self.run_test("bar >: foo", "(foo (bar 1) (bar 2))")
        self.run_test("bar >: foo", "(foo)")
        self.run_test("bar >: foo", "(foo (baz (bar)))")
        self.run_test("bar >: foo", "(bar (foo 1))")

        self.run_test("/.*/ >: foo", "(a (foo (bar 1)) (foo (baz 2)))", "(bar 1)", "(baz 2)")

    def test_preceding_following_sister(self):
        # test preceding sisters
        preceding = TregexPattern("/.*/ $.. baz")
        self.run_test(preceding, "(a (foo 1) (bar 2) (baz 3))", "(foo 1)", "(bar 2)")
        self.run_test(
            preceding, "(root (b (foo 1)) (a (foo 1) (bar 2) (baz 3)))", "(foo 1)", "(bar 2)"
        )
        self.run_test(
            preceding, "(root (a (foo 1) (bar 2) (baz 3)) (b (foo 1)))", "(foo 1)", "(bar 2)"
        )
        self.run_test(preceding, "(a (foo 1) (baz 2) (bar 3))", "(foo 1)")
        self.run_test(preceding, "(a (baz 1) (foo 2) (bar 3))")

        # test immediately preceding sisters
        impreceding = TregexPattern("/.*/ $. baz")
        self.run_test(impreceding, "(a (foo 1) (bar 2) (baz 3))", "(bar 2)")
        self.run_test(impreceding, "(root (b (foo 1)) (a (foo 1) (bar 2) (baz 3)))", "(bar 2)")
        self.run_test(impreceding, "(root (a (foo 1) (bar 2) (baz 3)) (b (foo 1)))", "(bar 2)")
        self.run_test(impreceding, "(a (foo 1) (baz 2) (bar 3))", "(foo 1)")
        self.run_test(impreceding, "(a (baz 1) (foo 2) (bar 3))")

        # test following sisters
        following = TregexPattern("/.*/ $,, baz")

        self.run_test(following, "(a (foo 1) (bar 2) (baz 3))")
        self.run_test(following, "(root (b (foo 1)) (a (foo 1) (bar 2) (baz 3)))")
        self.run_test(following, "(root (a (foo 1) (bar 2) (baz 3)) (b (foo 1)))")
        self.run_test(
            following, "(root (a (baz 1) (bar 2) (foo 3)) (b (foo 1)))", "(bar 2)", "(foo 3)"
        )
        self.run_test(following, "(a (foo 1) (baz 2) (bar 3))", "(bar 3)")
        self.run_test(following, "(a (baz 1) (foo 2) (bar 3))", "(foo 2)", "(bar 3)")

        # test immediately following sisters
        imfollowing = TregexPattern("/.*/ $, baz")
        self.run_test(imfollowing, "(a (foo 1) (bar 2) (baz 3))")
        self.run_test(imfollowing, "(root (b (foo 1)) (a (foo 1) (bar 2) (baz 3)))")
        self.run_test(imfollowing, "(root (a (foo 1) (bar 2) (baz 3)) (b (foo 1)))")
        self.run_test(imfollowing, "(root (a (baz 1) (bar 2) (foo 3)) (b (foo 1)))", "(bar 2)")
        self.run_test(imfollowing, "(a (foo 1) (baz 2) (bar 3))", "(bar 3)")
        self.run_test(imfollowing, "(a (baz 1) (foo 2) (bar 3))", "(foo 2)")

    def test_dominate_unary_chain(self):
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar))))))", "(foo (b (c (d (bar)))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar) (baz))))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar)) (baz)))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar))) (baz))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar)))) (baz)))")
        self.run_test(
            "foo <<: bar", "(a (foo (b (c (d (bar))))) (baz))", "(foo (b (c (d (bar)))))"
        )
        self.run_test("foo <<: bar", "(a (foo (b (c (bar)))))", "(foo (b (c (bar))))")
        self.run_test("foo <<: bar", "(a (foo (b (bar))))", "(foo (b (bar)))")
        self.run_test("foo <<: bar", "(a (foo (bar)))", "(foo (bar))")

        self.run_test("bar >>: foo", "(a (foo (b (c (d (bar))))))", "(bar)")
        self.run_test("bar >>: foo", "(a (foo (b (c (d (bar) (baz))))))")
        self.run_test("bar >>: foo", "(a (foo (b (c (d (bar)) (baz)))))")
        self.run_test("bar >>: foo", "(a (foo (b (c (d (bar))) (baz))))")
        self.run_test("bar >>: foo", "(a (foo (b (c (d (bar)))) (baz)))")
        self.run_test("bar >>: foo", "(a (foo (b (c (d (bar))))) (baz))", "(bar)")
        self.run_test("bar >>: foo", "(a (foo (b (c (bar)))))", "(bar)")
        self.run_test("bar >>: foo", "(a (foo (b (bar))))", "(bar)")
        self.run_test("bar >>: foo", "(a (foo (bar)))", "(bar)")

    def test_precedes_described_chain(self):
        self.run_test(
            "DT .+(JJ) NN", "(NP (DT the) (JJ large) (JJ green) (NN house))", "(DT the)"
        )
        self.run_test(
            "DT .+(@JJ) /^NN/",
            "(NP (PDT both) (DT the) (JJ-SIZE large) (JJ-COLOUR green) (NNS houses))",
            "(DT the)",
        )
        self.run_test(
            "NN ,+(JJ) DT", "(NP (DT the) (JJ large) (JJ green) (NN house))", "(NN house)"
        )
        self.run_test(
            "NNS ,+(@JJ) /^DT/",
            "(NP (PDT both) (DT the) (JJ-SIZE large) (JJ-COLOUR green) (NNS houses))",
            "(NNS houses)",
        )
        self.run_test(
            "NNS ,+(/^(JJ|DT).*$/) PDT",
            "(NP (PDT both) (DT the) (JJ-SIZE large) (JJ-COLOUR green) (NNS houses))",
            "(NNS houses)",
        )
        self.run_test(
            "NNS ,+(@JJ) JJ",
            "(NP (PDT both) (DT the) (JJ large) (JJ-COLOUR green) (NNS houses))",
            "(NNS houses)",
        )

    def test_dominate_described_chain(self):
        self.run_test("foo <+(bar) baz", "(a (foo (baz)))", "(foo (baz))")
        self.run_test("foo <+(bar) baz", "(a (foo (bar (baz))))", "(foo (bar (baz)))")
        self.run_test(
            "foo <+(bar) baz", "(a (foo (bar (bar (baz)))))", "(foo (bar (bar (baz))))"
        )
        self.run_test("foo <+(bar) baz", "(a (foo (bif (baz))))")
        self.run_test("foo <+(!bif) baz", "(a (foo (bif (baz))))")
        self.run_test("foo <+(!bif) baz", "(a (foo (bar (baz))))", "(foo (bar (baz)))")
        self.run_test("foo <+(/b/) baz", "(a (foo (bif (baz))))", "(foo (bif (baz)))")
        self.run_test(
            "foo <+(/b/) baz", "(a (foo (bar (bif (baz)))))", "(foo (bar (bif (baz))))"
        )
        self.run_test(
            "foo <+(bar) baz",
            "(a (foo (bar (blah 1) (bar (baz)))))",
            "(foo (bar (blah 1) (bar (baz))))",
        )

        self.run_test("baz >+(bar) foo", "(a (foo (baz)))", "(baz)")
        self.run_test("baz >+(bar) foo", "(a (foo (bar (baz))))", "(baz)")
        self.run_test("baz >+(bar) foo", "(a (foo (bar (bar (baz)))))", "(baz)")
        self.run_test("baz >+(bar) foo", "(a (foo (bif (baz))))")
        self.run_test("baz >+(!bif) foo", "(a (foo (bif (baz))))")
        self.run_test("baz >+(!bif) foo", "(a (foo (bar (baz))))", "(baz)")
        self.run_test("baz >+(/b/) foo", "(a (foo (bif (baz))))", "(baz)")
        self.run_test("baz >+(/b/) foo", "(a (foo (bar (bif (baz)))))", "(baz)")
        self.run_test("baz >+(bar) foo", "(a (foo (bar (blah 1) (bar (baz)))))", "(baz)")

    def test_segmented_and_equals_expressions(self):
        self.run_test("foo : bar", "(a (foo) (bar))", "(foo)")
        self.run_test("foo : bar", "(a (foo))")
        self.run_test(
            "(foo << bar) : (foo << baz)", "(a (foo (bar 1)) (foo (baz 2)))", "(foo (bar 1))"
        )
        self.run_test(
            "(foo << bar) : (foo << baz)", "(a (foo (bar 1)) (foo (baz 2)))", "(foo (bar 1))"
        )
        self.run_test("(foo << bar) == (foo << baz)", "(a (foo (bar)) (foo (baz)))")
        self.run_test(
            "(foo << bar) : (foo << baz)", "(a (foo (bar) (baz)))", "(foo (bar) (baz))"
        )
        self.run_test(
            "(foo << bar) == (foo << baz)", "(a (foo (bar) (baz)))", "(foo (bar) (baz))"
        )
        self.run_test("(foo << bar) : (baz >> a)", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("(foo << bar) == (baz >> a)", "(a (foo (bar) (baz)))")

        self.run_test("foo == foo", "(a (foo (bar)))", "(foo (bar))")
        self.run_test("foo << bar == foo", "(a (foo (bar)) (foo (baz)))", "(foo (bar))")
        self.run_test("foo << bar == foo", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("foo << bar == foo << baz", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("foo << bar : (foo << baz)", "(a (foo (bar)) (foo (baz)))", "(foo (bar))")

    def test_complex(self):
        test_pattern = (
            "S < (NP=m1 $.. (VP < ((/VB/ < /^(am|are|is|was|were|'m|'re|'s|be)$/) $.. NP=m2)))"
        )
        test_tree = (
            "(S (NP (NP (DT The) (JJ next) (NN stop)) (PP (IN on) (NP (DT the) (NN itinerary))))"
            " (VP (VBD was) (NP (NP (NNP Chad)) (, ,) (SBAR (WHADVP (WRB where)) (S (NP (NNP"
            " Chen)) (VP (VBD dined) (PP (IN with) (NP (NP (NNP Chad) (POS 's)) (NNP President)"
            " (NNP Idris) (NNP Debi)))))))) (. .))"
        )
        self.run_test(test_pattern, "(ROOT " + test_tree + ")", test_tree)

        test_tree = (
            "(S (NP (NNP Chen) (NNP Shui) (HYPH -) (NNP bian)) (VP (VBZ is) (NP (NP (DT the) (JJ"
            " first) (NML (NNP ROC) (NN president))) (SBAR (S (ADVP (RB ever)) (VP (TO to) (VP"
            " (VB travel) (PP (IN to) (NP (JJ western) (NNP Africa))))))))) (. .))"
        )
        self.run_test(test_pattern, "(ROOT " + test_tree + ")", test_tree)

        test_tree = (
            "(ROOT (S (NP (PRP$ My) (NN dog)) (VP (VBZ is) (VP (VBG eating) (NP (DT a) (NN"
            " sausage)))) (. .)))"
        )
        self.run_test(test_pattern, test_tree)

        test_tree = (
            "(ROOT (S (NP (PRP He)) (VP (MD will) (VP (VB be) (ADVP (RB here) (RB soon)))) (."
            " .)))"
        )
        self.run_test(test_pattern, test_tree)

        test_pattern = "/^NP(?:-TMP|-ADV)?$/=m1 < (NP=m2 $- /^,$/ $-- NP=m3 !$ CC|CONJP)"
        test_tree = (
            "(ROOT (S (NP (NP (NP (NP (DT The) (NNP ROC) (POS 's)) (NN ambassador)) (PP (IN to)"
            " (NP (NNP Nicaragua)))) (, ,) (NP (NNP Antonio) (NNP Tsai)) (, ,)) (ADVP (RB"
            " bluntly)) (VP (VBD argued) (PP (IN in) (NP (NP (DT a) (NN briefing)) (PP (IN with)"
            " (NP (NNP Chen))))) (SBAR (IN that) (S (NP (NP (NP (NNP Taiwan) (POS 's)) (JJ"
            " foreign) (NN assistance)) (PP (IN to) (NP (NNP Nicaragua)))) (VP (VBD was) (VP"
            " (VBG being) (ADJP (JJ misused))))))) (. .)))"
        )
        expected_result = (
            "(NP (NP (NP (NP (DT The) (NNP ROC) (POS 's)) (NN ambassador)) (PP (IN to) (NP (NNP"
            " Nicaragua)))) (, ,) (NP (NNP Antonio) (NNP Tsai)) (, ,))"
        )
        self.run_test(test_pattern, test_tree, expected_result)

        test_tree = (
            "(ROOT (S (PP (IN In) (NP (NP (DT the) (NN opinion)) (PP (IN of) (NP (NP (NNP"
            " Norman) (NNP Hsu)) (, ,) (NP (NP (NN vice) (NN president)) (PP (IN of) (NP (NP (DT"
            " a) (NNS foods) (NN company)) (SBAR (WHNP (WHNP (WP$ whose) (NN family)) (PP (IN"
            " of) (NP (CD four)))) (S (VP (VBD had) (VP (VBN spent) (NP (QP (DT a) (JJ few))"
            " (NNS years)) (PP (IN in) (NP (NNP New) (NNP Zealand))) (PP (IN before) (S (VP (VBG"
            ' moving) (PP (IN to) (NP (NNP Dongguan))))))))))))))))) (, ,) (`` ") (NP (NP (DT'
            " The) (JJ first) (NN thing)) (VP (TO to) (VP (VB do)))) (VP (VBZ is) (S (VP (VB"
            " ask) (NP (DT the) (NNS children)) (NP (PRP$ their) (NN reason)) (PP (IN for) (S"
            " (VP (VBG saying) (NP (JJ such) (NNS things)))))))) (. .)))"
        )
        expected_result = (
            "(NP (NP (NNP Norman) (NNP Hsu)) (, ,) (NP (NP (NN vice) (NN president)) (PP (IN of)"
            " (NP (NP (DT a) (NNS foods) (NN company)) (SBAR (WHNP (WHNP (WP$ whose) (NN"
            " family)) (PP (IN of) (NP (CD four)))) (S (VP (VBD had) (VP (VBN spent) (NP (QP (DT"
            " a) (JJ few)) (NNS years)) (PP (IN in) (NP (NNP New) (NNP Zealand))) (PP (IN"
            " before) (S (VP (VBG moving) (PP (IN to) (NP (NNP Dongguan))))))))))))))"
        )
        self.run_test(test_pattern, test_tree, expected_result)

        test_tree = (
            "(ROOT (S (NP (NP (NNP Banana)) (, ,) (NP (NN orange)) (, ,) (CC and) (NP (NN"
            " apple))) (VP (VBP are) (NP (NNS fruits))) (. .)))"
        )
        self.run_test(test_pattern, test_tree)

        test_tree = (
            "(ROOT (S (NP (PRP He)) (, ,) (ADVP (RB however)) (, ,) (VP (VBZ does) (RB not) (VP"
            " (VB look) (ADJP (JJ fine)))) (. .)))"
        )
        self.run_test(test_pattern, test_tree)

    def test_complex2(self):
        inputTrees = [
            (
                "(ROOT (S (NP (PRP You)) (VP (VBD did) (VP (VB go) (WHADVP (WRB How) (JJ long))"
                " (PP (IN for)))) (. .)))"
            ),
            (
                "(ROOT (S (NP (NNS Raccoons)) (VP (VBP do) (VP (VB come) (WHADVP (WRB When))"
                " (PRT (RP out)))) (. .)))"
            ),
            (
                "(ROOT (S (NP (PRP She)) (VP (VBZ is) (VP (WHADVP (WRB Where)) (VBG working)))"
                " (. .)))"
            ),
            "(ROOT (S (NP (PRP You)) (VP (VBD did) (VP (WHNP (WP What)) (VB do))) (. .)))",
            (
                "(ROOT (S (NP (PRP You)) (VP (VBD did) (VP (VB do) (PP (IN in) (NP (NNP"
                " Australia))) (WHNP (WP What)))) (. .)))"
            ),
        ]

        # TODO
        # pattern = "WHADVP=whadvp > VP $+ /[A-Z]*/=last ![$++ (PP < NP)]"
        # self.run_test(pattern, inputTrees[0], "(WHADVP (WRB How) (JJ long))")
        # self.run_test(pattern, inputTrees[1], "(WHADVP (WRB When))")
        # self.run_test(pattern, inputTrees[2], "(WHADVP (WRB Where))")
        # self.run_test(pattern, inputTrees[3])
        # self.run_test(pattern, inputTrees[4])

        pattern = "VP < (/^WH/=wh $++ /^VB/=vb)"
        self.run_test(pattern, inputTrees[0])
        self.run_test(pattern, inputTrees[1])
        self.run_test(pattern, inputTrees[2], "(VP (WHADVP (WRB Where)) (VBG working))")
        self.run_test(pattern, inputTrees[3], "(VP (WHNP (WP What)) (VB do))")
        self.run_test(pattern, inputTrees[4])

        pattern = "PP=pp > VP $+ WHNP=whnp"
        self.run_test(pattern, inputTrees[0])
        self.run_test(pattern, inputTrees[1])
        self.run_test(pattern, inputTrees[2])
        self.run_test(pattern, inputTrees[3])
        self.run_test(pattern, inputTrees[4], "(PP (IN in) (NP (NNP Australia)))")

    def test_ancestor_of_ith_leaf(self):
        self.run_test("A <<<1 b", "(ROOT (A (B b)))", "(A (B b))")
        self.run_test("A <<<2 b", "(ROOT (A (B b)))")
        self.run_test("A <<<-1 b", "(ROOT (A (B b)))", "(A (B b))")
        self.run_test("A <<<1 b", "(ROOT (A (B z) (C b)))")
        self.run_test("A <<<2 b", "(ROOT (A (B z) (C b)))", "(A (B z) (C b))")
        self.run_test("A <<<-1 b", "(ROOT (A (B z) (C b)))", "(A (B z) (C b))")
        self.run_test("A <<<-2 b", "(ROOT (A (B z) (C b)))")
        self.run_test("A <<<-1 z", "(ROOT (A (B z) (C b)))")
        self.run_test("A <<<-2 z", "(ROOT (A (B z) (C b)))", "(A (B z) (C b))")

    def test_head_of_phrase(self):
        self.run_test(
            "NP <# NNS", "(NP (NN work) (NNS practices))", "(NP (NN work) (NNS practices))"
        )
        self.run_test("NP <# NN", "(NP (NN work) (NNS practices))")
        # should have no results
        self.run_test(
            "NP <<# NNS",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
            "(NP (NN work) (NNS practices))",
        )
        self.run_test(
            "NP !<# NNS <<# NNS",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
        )
        self.run_test(
            "NP !<# NNP <<# NNP",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
        )
        # no results
        self.run_test(
            "NNS ># NP",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
            "(NNS practices)",
        )
        self.run_test(
            "NNS ># (NP < PP)",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
        )
        # no results
        self.run_test(
            "NNS >># (NP < PP)",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
            "(NNS practices)",
        )
        self.run_test(
            "NP <<# /^NN/",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
            "(NP (NP (NN work) (NNS practices)) (PP (IN in) (NP (DT the) (JJ former) (NNP"
            " Soviet) (NNP Union))))",
            "(NP (NN work) (NNS practices))",
            "(NP (DT the) (JJ former) (NNP Soviet) (NNP Union))",
        )

    def test_chinese(self):
        """
        Test a pattern with chinese characters in it, just to make sure
        that also works
        """
        pattern = TregexPattern("DEG|DEC < 的")
        self.run_test("DEG|DEC < 的", "(DEG (的 1))", "(DEG (的 1))")

    def test_parenthesized_expressions(self):
        tree_strings = [
            (
                "( (S (S (PP (IN In) (NP (CD 1941) )) (, ,) (NP (NP (NNP Raeder) ) (CC and) (NP"
                " (DT the) (JJ German) (NN navy) )) (VP (VBD threatened) (S (VP (TO to) (VP (VB"
                " attack) (NP (DT the) (NNP Panama) (NNP Canal) )))))) (, ,) (RB so) (S (NP (PRP"
                " we) ) (VP (VBD created) (NP (NP (DT the) (NNP Southern) (NNP Command) )"
                " (PP-LOC (IN in) (NP (NNP Panama) ))))) (. .) ))"
            ),
            (
                "(S (S (NP-SBJ (NNP Japan) ) (VP (MD can) (VP (VP (VB grow) ) (CC and) (VP (RB"
                " not) (VB cut) (PRT (RB back) ))))) (, ,) (CC and) (RB so) (S (ADVP (RB too) )"
                " (, ,) (NP (NP (NNP New) (NNP Zealand) )) ))"
            ),
            (
                "( (S (S (NP-SBJ (PRP You) ) (VP (VBP make) (NP (DT a) (NN forecast) ))) (, ,)"
                " (CC and) (RB then) (S (NP-SBJ (PRP you) ) (VP (VBP become) (NP-PRD (PRP$ its)"
                " (NN prisoner) ))) (. .)))"
            ),
        ]

        # First pattern: no parenthesized expressions.  All three trees should match once.
        pattern = TregexPattern("/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ $+ (RB=adv $+ /^S/)))")
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(1, len(matches))

        # Second pattern: single relation in parentheses.  First tree should not match.
        pattern = TregexPattern(
            "/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ (< and) $+ (RB=adv $+ /^S/)))"
        )
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(0, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(1, len(matches))

        # Third pattern: single relation in parentheses and negated.  Only first tree should match.
        pattern = TregexPattern(
            "/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ !(< and) $+ (RB=adv $+ /^S/)))"
        )
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(0, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(0, len(matches))

        # Fourth pattern: double relation in parentheses, no negation.
        pattern = TregexPattern(
            "/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ (< and $+ RB) $+ (RB=adv $+ /^S/)))"
        )
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(0, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(1, len(matches))

        # Fifth pattern: double relation in parentheses, negated.
        pattern = TregexPattern(
            "/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ !(< and $+ RB) $+ (RB=adv $+ /^S/)))"
        )
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(0, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(0, len(matches))

        # Six pattern: double relation in parentheses, negated.  The only
        # tree with "and then" is the third one, so that is the one tree
        # that should not match.
        pattern = TregexPattern(
            "/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ !(< and $+ (RB < then)) $+ (RB=adv $+ /^S/)))"
        )
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(0, len(matches))

    def test_parent_equals(self):
        """
        The PARENT_EQUALS relation allows for a simplification of what
        would have been a pair of rules in the dependencies.
        """
        self.run_test("A <= B", "(A (B 1))", "(A (B 1))")
        # Note that if the child node is the same as the parent node, a
        # double match is expected if there is nothing to eliminate it in
        # the expression
        self.run_test("A <= A", "(A (A 1) (B 2))", "(A (A 1) (B 2))", "(A (A 1) (B 2))", "(A 1)")
        # This is the kind of expression where this relation can be useful
        self.run_test("A <= (A < B)", "(A (A (B 1)))", "(A (A (B 1)))", "(A (B 1))")
        self.run_test(
            "A <= (A < B)", "(A (A (B 1)) (A (C 2)))", "(A (A (B 1)) (A (C 2)))", "(A (B 1))"
        )
        self.run_test("A <= (A < B)", "(A (A (C 2)))")

    def test_root_disjunction(self):
        """
        Test a few possible ways to make disjunctions at the root level.
        Note that disjunctions at lower levels can always be created by
        repeating the relation, but that is not true at the root, since
        the root "relation" is implicit.
        """
        self.run_test("A ; B", "(A (B 1))", "(A (B 1))", "(B 1)")

        self.run_test("(A) ; (B)", "(A (B 1))", "(A (B 1))", "(B 1)")
        self.run_test("(A|C) ; (B)", "(A (B 1))", "(A (B 1))", "(B 1)")

        self.run_test("A < B ; A < C", "(A (B 1) (C 2))", "(A (B 1) (C 2))", "(A (B 1) (C 2))")

        self.run_test("A < B ; B < C", "(A (B 1) (C 2))", "(A (B 1) (C 2))")
        self.run_test("A < B ; B < C", "(A (B 1) (C 2))", "(A (B 1) (C 2))")
        self.run_test("A < B ; B < C", "(A (B (C 1)) (C 2))", "(A (B (C 1)) (C 2))", "(B (C 1))")

        self.run_test(
            "A ; B ; C",
            "(A (B (C 1)) (C 2))",
            "(A (B (C 1)) (C 2))",
            "(B (C 1))",
            "(C 1)",
            "(C 2)",
        )

        # The binding of the | should look like this:
        # A ( (< B) | (< C) )

        self.run_test("A < B || < C", "(A (B 1))", "(A (B 1))")
        self.run_test("A < B || < C", "(A (B 1) (C 2))", "(A (B 1) (C 2))", "(A (B 1) (C 2))")
        self.run_test("A < B || < C", "(B (C 1))")

    def test_subtree_pattern(self):
        """
        Tests the subtree pattern, <..., which checks for
        an exact subtree under our current tree
        """
        # test the obvious expected matches and several expected match failures
        self.run_test("A <... { B ; C ; D }", "(A (B 1) (C 2) (D 3))", "(A (B 1) (C 2) (D 3))")
        self.run_test(
            "A <... { B ; C ; D }", "(Z (A (B 1) (C 2) (D 3)))", "(A (B 1) (C 2) (D 3))"
        )
        self.run_test("A <... { B ; C ; D }", "(A (B 1) (C 2) (D 3) (E 4))")
        self.run_test("A <... { B ; C ; D }", "(A (E 4) (B 1) (C 2) (D 3))")
        self.run_test("A <... { B ; C ; D }", "(A (B 1) (C 2) (E 4) (D 3))")
        self.run_test("A <... { B ; C ; D }", "(A (B 1) (C 2))")

        # every test above should return the opposite when negated
        self.run_test("A !<... { B ; C ; D }", "(A (B 1) (C 2) (D 3))")
        self.run_test("A !<... { B ; C ; D }", "(Z (A (B 1) (C 2) (D 3)))")
        self.run_test(
            "A !<... { B ; C ; D }", "(A (B 1) (C 2) (D 3) (E 4))", "(A (B 1) (C 2) (D 3) (E 4))"
        )
        self.run_test(
            "A !<... { B ; C ; D }", "(A (E 4) (B 1) (C 2) (D 3))", "(A (E 4) (B 1) (C 2) (D 3))"
        )
        self.run_test(
            "A !<... { B ; C ; D }", "(A (B 1) (C 2) (E 4) (D 3))", "(A (B 1) (C 2) (E 4) (D 3))"
        )
        self.run_test("A !<... { B ; C ; D }", "(A (B 1) (C 2))", "(A (B 1) (C 2))")

        # test a couple various forms of nesting
        self.run_test("A <... { (B < C) ; D }", "(A (B (C 2)) (D 3))", "(A (B (C 2)) (D 3))")
        self.run_test(
            "A <... { (B <... { C ; D }) ; E }",
            "(A (B (C 2) (D 3)) (E 4))",
            "(A (B (C 2) (D 3)) (E 4))",
        )
        self.run_test("A <... { (B !< C) ; D }", "(A (B (C 2)) (D 3))")

    def test_disjunction_variable_assignments(self):
        tree_string = (
            "(NP (UCP (NNP U.S.) (CC and) (ADJP (JJ northern) (JJ European))) (NNS diplomats))"
        )
        pattern = TregexPattern("UCP [ <- (ADJP=adjp < JJR) || <, NNP=np ]")

        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))

        handled_nodes = pattern.get_nodes("np")
        self.assertIsNotNone(handled_nodes)
        self.assertEqual(1, len(handled_nodes))
        self.assertEqual("(NNP U.S.)", handled_nodes[0].to_string())

    def test_optional(self):
        tree_string = "(A (B (C 1)) (B 2))"
        pattern = TregexPattern("B ? < C=c")

        matches = pattern.findall(tree_string)
        self.assertEqual(2, len(matches))

        handled_nodes = pattern.get_nodes("c")
        self.assertIsNotNone(handled_nodes)
        self.assertEqual(1, len(handled_nodes))
        self.assertEqual("(C 1)", handled_nodes[0].to_string())

        # TODO =punc
        # tree_string = "(ROOT (INTJ (CC But) (S (NP (DT the) (NNP RTC)) (ADVP (RB also)) (VP (VBZ requires) (`` ``) (S (FRAG (VBG working) ('' '') (NP (NP (NN capital)) (S (VP (TO to) (VP (VB maintain) (SBAR (S (NP (NP (DT the) (JJ bad) (NNS assets)) (PP (IN of) (NP (NP (NNS thrifts)) (SBAR (WHNP (WDT that)) (S (VP (VBP are) (VBN sold) (, ,) (PP (IN until) (NP (DT the) (NNS assets))))))))) (VP (MD can) (VP (VB be) (VP (VBN sold) (ADVP (RB separately))))))))))))))) (S (VP (. .)))))"
        # # a pattern used to rearrange punctuation nodes in the srparser
        # pattern = TregexPattern("__ !> __ <- (__=top <- (__ <<- (/[.]|PU/=punc < /[.!?。！？]/ ?> (__=single <: =punc))))")
        #
        # matches = pattern.findall(tree_string)
        # self.assertEqual(1, len(matches))
        #
        # handled_nodes = pattern.get_nodes("punc")
        # self.assertEqual("(. .)", handled_nodes[0].to_string())
        # self.assertEqual("(VP (. .))", handled_nodes[1].to_string())

    # TODO
    # def test_negated_disjunction(self):
    #     """
    #     A user supplied an example of a negated disjunction which went into an infinite loop.
    #     Apparently no one had ever used a negated disjunction of tree structures before!
    #     <br>
    #     The problem was that the logic at the time tried to backtrack in
    #     the disjunction to find a better match, but that resulted in it
    #     going back and forth between the failed clause which was accepted
    #     and the successful clause which was rejected.  The problem being
    #     that the first half of the disjunction doesn't match, so the
    #     pattern is successful up to that point, but the second half does
    #     match, causing the pattern to be rejected and restarted.
    #     """
    #     self.run_test("NP![</,/|.(JJ<else)]", "( (NP (NP (NN anyone)) (ADJP (JJ else))))", "(NP (NP (NN anyone)) (ADJP (JJ else)))")

    def run_test(
        self, pattern: Union[TregexPattern, str], tree_str: str, *expected_results: str
    ):
        """
        Check that running the Tregex pattern on the tree gives the results
        shown in results.
        """
        # bad: pattern has two possible types
        if isinstance(pattern, str):
            pattern = TregexPattern(pattern)
        matches = pattern.findall(tree_str)
        self.assertEqual(len(matches), len(expected_results))

        for match, expected_result in zip(matches, expected_results):
            # Tree.from_string returns a Tree list of length 1 for single-tree input
            expected_tree = Tree.from_string(expected_result)[0]
            self.assertEqual(match.to_string(), expected_tree.to_string())
