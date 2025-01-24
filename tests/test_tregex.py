#!/usr/bin/env python3

# translated from https://github.com/stanfordnlp/CoreNLP/blob/efc66a9cf49fecba219dfaa4025315ad966285cc/test/src/edu/stanford/nlp/trees/tregex/TregexTest.java
# last modified at Apr 2, 2022 (https://github.com/stanfordnlp/CoreNLP/commits/efc66a9cf49fecba219dfaa4025315ad966285cc/test/src/edu/stanford/nlp/trees/tregex/TregexTest.java)

from typing import Union

from pytregex.exceptions import ParseException
from pytregex.tree import Tree
from pytregex.tregex import TregexPattern

from .base_tmpl import BaseTmpl
from .base_tmpl import tree as tree_string


class TestTregex(BaseTmpl):
    def setUp(self):
        self.tree_string = tree_string
        return super().setUp()

    def test_JoãoSilva(self):
        tregex1 = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, __=m <- (__ <, CONJ <- __=z))))")
        tregex2 = TregexPattern(
            "PNT=p >>- (/(.+)/#1%var=l >, (__=t <- (__=r <, /(.+)/#1%var=m <- (__ <, CONJ <- /(.+)/#1%var=z))))"
        )
        tregex3 = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, ~l <- (__ <, CONJ <- ~l))))")
        tree_string = "(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))"

        self.assertTrue(tregex1.findall(tree_string))
        # self.assertTrue(tregex2.findall(tree_string))
        # self.assertTrue(tregex3.findall(tree_string))

    def test_no_results(self):
        pMWE = TregexPattern("/^MW/")
        self.assertFalse(pMWE.findall("(Foo)"))

    def test_one_result(self):
        pMWE = TregexPattern("/^MW/")
        matches = pMWE.findall("(ROOT (MWE (N 1) (N 2) (N 3)))")
        self.assertEqual(1, len(matches))
        self.assertEqual("(MWE (N 1) (N 2) (N 3))", matches[0].tostring())

    def test_two_results(self):
        pMWE = TregexPattern("/^MW/")
        matches = pMWE.findall("(ROOT (MWE (N 1) (N 2) (N 3)) (MWV (A B)))")
        self.assertEqual(2, len(matches))

        self.assertEqual("(MWE (N 1) (N 2) (N 3))", matches[0].tostring())
        self.assertEqual("(MWV (A B))", matches[1].tostring())

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
        self.run_test("/.*/ >-1 root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(b (baz 5))")
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
        self.run_test(dominates_pattern, "(foo (a (b 1) (c 2) (bar 3)))", "(foo (a (b 1) (c 2) (bar 3)))")
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
        self.run_test(dominated_pattern, "(a (foo bar))")
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

        pattern = TregexPattern("foo $ foo=a")
        matches = pattern.findall("(a (foo 1) (foo 2) (foo 3))")
        nodes = pattern.get_nodes("a")
        self.assertEqual(6, len(matches))

        self.assertEqual("(foo 1)", matches[0].tostring())
        self.assertEqual("(foo 2)", nodes[0].tostring())

        self.assertEqual("(foo 1)", matches[1].tostring())
        self.assertEqual("(foo 3)", nodes[1].tostring())

        self.assertEqual("(foo 2)", matches[2].tostring())
        self.assertEqual("(foo 1)", nodes[2].tostring())

        self.assertEqual("(foo 2)", matches[3].tostring())
        self.assertEqual("(foo 3)", nodes[3].tostring())

        self.assertEqual("(foo 3)", matches[4].tostring())
        self.assertEqual("(foo 1)", nodes[4].tostring())

        self.assertEqual("(foo 3)", matches[5].tostring())
        self.assertEqual("(foo 2)", nodes[5].tostring())

        self.run_test("foo $ foo", "(a (foo 1))")

    def test_precedes_follows(self):
        # precedes
        pattern = TregexPattern("/.*/ .. foo")

        self.run_test(pattern, "(a (foo 1) (bar 2))")
        self.run_test(pattern, "(a (bar 1) (foo 2))", "(bar 1)", "(1)")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo 3))", "(bar 1)", "(1)", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (foo 1) (baz 2) (bar 3))")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2))")
        self.run_test(pattern, "(a (bar 1) (baz (foo 2)))", "(bar 1)", "(1)")
        self.run_test(pattern, "(a (bar 1) (baz 2) (bif (foo 3)))", "(bar 1)", "(1)", "(baz 2)", "(2)")
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2) (bif 3))")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo (bif 3)))", "(bar 1)", "(1)", "(baz 2)", "(2)")
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
        self.run_test(pattern, "(a (bar (foo 1)) (baz 2) (bif 3))", "(baz 2)", "(2)", "(bif 3)", "(3)")
        self.run_test(pattern, "(a (bar 1) (baz 2) (foo (bif 3)))")
        self.run_test(pattern, "(a (foo (bif 1)) (bar 2) (baz 3))", "(bar 2)", "(2)", "(baz 3)", "(3)")
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
        self.run_test(pattern, "(a (bar 1) (foo 2) (foo 3) (baz 4))", "(bar 1)", "(1)", "(foo 2)", "(2)")
        self.run_test(pattern, "(a (b (c 1) (d 2)) (foo))", "(b (c 1) (d 2))", "(d 2)", "(2)")
        self.run_test(pattern, "(a (b (c 1) (d 2)) (bar (foo 3)))", "(b (c 1) (d 2))", "(d 2)", "(2)")
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
        self.run_test(pattern, "(a (bar 1) (foo 2) (foo 3) (baz 4))", "(foo 3)", "(3)", "(baz 4)", "(4)")
        self.run_test(pattern, "(a (foo) (b (c 1) (d 2)))", "(b (c 1) (d 2))", "(c 1)", "(1)")
        self.run_test(pattern, "(a (bar (foo 3)) (b (c 1) (d 2)))", "(b (c 1) (d 2))", "(c 1)", "(1)")
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
        self.run_test("/.*/ <<, baz", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(b (baz 5))")

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
        self.run_test("/.*/ >>, a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(foo 1 2)", "(1)")
        self.run_test("/.*/ >>, bar", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(3)")

        # A rightmost descendant of B
        self.run_test(
            "/.*/ >>- root",
            "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))",
            "(b (baz 5))",
            "(baz 5)",
            "(5)",
        )
        self.run_test("/.*/ >>- a", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(bar 3 4)", "(4)")
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
        self.run_test("/.*/ >- root", "(root (a (foo 1 2) (bar 3 4)) (b (baz 5)))", "(b (baz 5))")
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
        self.run_test(preceding, "(root (b (foo 1)) (a (foo 1) (bar 2) (baz 3)))", "(foo 1)", "(bar 2)")
        self.run_test(preceding, "(root (a (foo 1) (bar 2) (baz 3)) (b (foo 1)))", "(foo 1)", "(bar 2)")
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
        self.run_test(following, "(root (a (baz 1) (bar 2) (foo 3)) (b (foo 1)))", "(bar 2)", "(foo 3)")
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

    # TODO
    #     def test_category_functions(self):
    # Function<String, String> fooCategory = new Function<String, String>()
    #   public String apply(String label)
    #     if (label == null)
    #       return label
    #
    #     if (label.equals("bar"))
    #       return "foo"
    #
    #     return label
    #
    #
    # TregexPatternCompiler fooCompiler = new TregexPatternCompiler(fooCategory)
    #
    # TregexPattern fooTregex = fooCompiler("@foo > bar")
    #         self.run_test(fooTregex, "(bar (foo 0))", "(foo 0)")
    #         self.run_test(fooTregex, "(bar (bar 0))", "(bar 0)")
    #         self.run_test(fooTregex, "(foo (foo 0))")
    #         self.run_test(fooTregex, "(foo (bar 0))")
    #
    # Function<String, String> barCategory = new Function<String, String>()
    #   public String apply(String label)
    #     if (label == null)
    #       return label
    #
    #     if (label.equals("foo"))
    #       return "bar"
    #
    #     return label
    #
    #
    # TregexPatternCompiler barCompiler = new TregexPatternCompiler(barCategory)
    #
    # TregexPattern barTregex = barCompiler("@bar > foo")
    #         self.run_test(barTregex, "(bar (foo 0))")
    #         self.run_test(barTregex, "(bar (bar 0))")
    #         self.run_test(barTregex, "(foo (foo 0))", "(foo 0)")
    #         self.run_test(barTregex, "(foo (bar 0))", "(bar 0)")
    #
    # # These should still work, since the tregex patterns have
    # # different category functions.  Old enough versions of tregex do
    # # not allow for that.
    #         self.run_test(fooTregex, "(bar (foo 0))", "(foo 0)")
    #         self.run_test(fooTregex, "(bar (bar 0))", "(bar 0)")
    #         self.run_test(fooTregex, "(foo (foo 0))")
    #         self.run_test(fooTregex, "(foo (bar 0))")

    def test_dominate_unary_chain(self):
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar))))))", "(foo (b (c (d (bar)))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar) (baz))))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar)) (baz)))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar))) (baz))))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar)))) (baz)))")
        self.run_test("foo <<: bar", "(a (foo (b (c (d (bar))))) (baz))", "(foo (b (c (d (bar)))))")
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
        self.run_test("DT .+(JJ) NN", "(NP (DT the) (JJ large) (JJ green) (NN house))", "(DT the)")
        self.run_test(
            "DT .+(@JJ) /^NN/",
            "(NP (PDT both) (DT the) (JJ-SIZE large) (JJ-COLOUR green) (NNS houses))",
            "(DT the)",
        )
        self.run_test("NN ,+(JJ) DT", "(NP (DT the) (JJ large) (JJ green) (NN house))", "(NN house)")
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
        self.run_test("foo <+(bar) baz", "(a (foo (bar (bar (baz)))))", "(foo (bar (bar (baz))))")
        self.run_test("foo <+(bar) baz", "(a (foo (bif (baz))))")
        self.run_test("foo <+(!bif) baz", "(a (foo (bif (baz))))")
        self.run_test("foo <+(!bif) baz", "(a (foo (bar (baz))))", "(foo (bar (baz)))")
        self.run_test("foo <+(/b/) baz", "(a (foo (bif (baz))))", "(foo (bif (baz)))")
        self.run_test("foo <+(/b/) baz", "(a (foo (bar (bif (baz)))))", "(foo (bar (bif (baz))))")
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
        self.run_test("(foo << bar) : (foo << baz)", "(a (foo (bar 1)) (foo (baz 2)))", "(foo (bar 1))")
        self.run_test("(foo << ban) $++ (foo << baz)", "(a (foo (bar 1)) (foo (baz 2)))")
        self.run_test("(foo << bar) == (foo << baz)", "(a (foo (bar)) (foo (baz)))")
        self.run_test("(foo << bar) : (foo << baz)", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("(foo << bar) == (foo << baz)", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("(foo << bar) : (baz >> a)", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("(foo << bar) == (baz >> a)", "(a (foo (bar) (baz)))")

        self.run_test("foo == foo", "(a (foo (bar)))", "(foo (bar))")
        self.run_test("foo << bar == foo", "(a (foo (bar)) (foo (baz)))", "(foo (bar))")
        self.run_test("foo << bar == foo", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("foo << bar == foo << baz", "(a (foo (bar) (baz)))", "(foo (bar) (baz))")
        self.run_test("foo << bar : (foo << baz)", "(a (foo (bar)) (foo (baz)))", "(foo (bar))")

    def test_two_children(self):
        self.run_test("foo << bar << baz", "(foo (bar 1) (baz 2))", "(foo (bar 1) (baz 2))")
        # this is a poorly written pattern that will match 4 times
        self.run_test(
            "foo << __ << baz",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
        )
        # this one also matches 4 times
        self.run_test(
            "foo << bar << __",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
        )
        # this one also matches 4 times
        self.run_test(
            "foo << __ << __",
            "(foo (bar 1))",
            "(foo (bar 1))",
            "(foo (bar 1))",
            "(foo (bar 1))",
            "(foo (bar 1))",
        )
        # same thing, just making sure variable assignment doesn't throw
        # it off
        self.run_test(
            "foo << __=a << __=b",
            "(foo (bar 1))",
            "(foo (bar 1))",
            "(foo (bar 1))",
            "(foo (bar 1))",
            "(foo (bar 1))",
        )
        # 16 times!  hopefully no one writes patterns like this
        self.run_test(
            "foo << __ << __",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
            "(foo (bar 1) (baz 2))",
        )
        # note: this matches because we set a=(bar 1), b=(1)
        # TODO
        # self.run_test("(foo << __=a << __=b) : (=a !== =b)", "(foo (bar 1))",
        # "(foo (bar 1))", "(foo (bar 1))")
        # self.run_test("(foo < __=a < __=b) : (=a !== =b)", "(foo (bar 1))")

        # TODO
        # 12 times: 16 possible ways to match the nodes, but 4 of them
        # are ruled out because they are the same node matching twice
        # self.run_test("(foo << __=a << __=b) : (=a !== =b)",
        # "(foo (bar 1) (baz 2))",
        # "(foo (bar 1) (baz 2))", "(foo (bar 1) (baz 2))",
        # "(foo (bar 1) (baz 2))", "(foo (bar 1) (baz 2))",
        # "(foo (bar 1) (baz 2))", "(foo (bar 1) (baz 2))",
        # "(foo (bar 1) (baz 2))", "(foo (bar 1) (baz 2))",
        # "(foo (bar 1) (baz 2))", "(foo (bar 1) (baz 2))",
        # "(foo (bar 1) (baz 2))", "(foo (bar 1) (baz 2))")

        # TODO
        # would need three unique descendants, but we only have two, so
        # this pattern doesn't match anything
        # self.run_test("(foo << __=a << __=b << __=c) : " +
        # "(=a !== =b) : (=a !== =c) : (=b !== =c)",
        # "(foo (bar 1))")

        # TODO: this should work, but it doesn't even parse
        # self.run_test("foo << __=a << __=b : !(=a == =b)", "(foo (bar 1))")

    def test_named(self):
        tree_string = "(a (foo 1) (bar 2) (bar 3))"
        pattern = TregexPattern("foo=a $ bar=b")
        matches = pattern.findall(tree_string)
        self.assertEqual(2, len(matches))
        self.assertEqual("(foo 1)", matches[0].tostring())
        self.assertEqual("(foo 1)", pattern.get_nodes("a")[0].tostring())
        self.assertEqual("(bar 2)", pattern.get_nodes("b")[0].tostring())

        self.assertEqual("(foo 1)", matches[1].tostring())
        self.assertEqual("(foo 1)", pattern.get_nodes("a")[1].tostring())
        self.assertEqual("(bar 3)", pattern.get_nodes("b")[1].tostring())

    def test_month_day_year(self):
        """
        An example from our code which looks for month-day-year patterns
        in PTB.  Relies on the pattern splitting and variable matching
        features.
        """
        MONTH_REGEX = "January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.|Feb\\.|Mar\\.|Apr\\.|Aug\\.|Sep\\.|Sept\\.|Oct\\.|Nov\\.|Dec\\."
        testPattern = (
            "NP=root <1 (NP=monthdayroot <1 (NNP=month <: /"
            + MONTH_REGEX
            + "/) <2 (CD=day <: __)) <2 (/^,$/=comma <: /^,$/) <3 (NP=yearroot <: (CD=year <:"
            " __)) : (=root <- =yearroot) : (=monthdayroot <- =day)"
        )

        # TODO
        # self.run_test(testPattern, "(ROOT (S (NP (NNP Mr.) (NNP Good)) (VP (VBZ devotes) (NP (RB much) (JJ serious) (NN space)) (PP (TO to) (NP (NP (DT the) (NNS events)) (PP (IN of) (NP (NP (NP (NNP Feb.) (CD 25)) (, ,) (NP (CD 1942))) (, ,) (SBAR (WHADVP (WRB when)) (S (NP (JJ American) (NNS gunners)) (VP (VBD spotted) (NP (NP (JJ strange) (NNS lights)) (PP (IN in) (NP (NP (DT the) (NN sky)) (PP (IN above) (NP (NNP Los) (NNP Angeles)))))))))))))) (. .)))", "(NP (NP (NNP Feb.) (CD 25)) (, ,) (NP (CD 1942)))")
        # self.run_test(testPattern, "(ROOT (S (NP (DT The) (JJ preferred) (NNS shares)) (VP (MD will) (VP (VB carry) (NP (NP (DT a) (JJ floating) (JJ annual) (NN dividend)) (ADJP (JJ equal) (PP (TO to) (NP (NP (CD 72) (NN %)) (PP (IN of) (NP (NP (DT the) (JJ 30-day) (NNS bankers) (POS ')) (NN acceptance) (NN rate))))))) (PP (IN until) (NP (NP (NNP Dec.) (CD 31)) (, ,) (NP (CD 1994)))))) (. .)))", "(NP (NP (NNP Dec.) (CD 31)) (, ,) (NP (CD 1994)))")
        # self.run_test(testPattern, "(ROOT (S (NP (PRP It)) (VP (VBD said) (SBAR (S (NP (NN debt)) (VP (VBD remained) (PP (IN at) (NP (NP (DT the) (QP ($ $) (CD 1.22) (CD billion))) (SBAR (WHNP (DT that)) (S (VP (VBZ has) (VP (VBD prevailed) (PP (IN since) (NP (JJ early) (CD 1989))))))))) (, ,) (SBAR (IN although) (S (NP (IN that)) (VP (VBN compared) (PP (IN with) (NP (NP (QP ($ $) (CD 911) (CD million))) (PP (IN at) (NP (NP (NNP Sept.) (CD 30)) (, ,) (NP (CD 1988))))))))))))) (. .)))", "(NP (NP (NNP Sept.) (CD 30)) (, ,) (NP (CD 1988)))")
        # self.run_test(testPattern, "(ROOT (S (NP (DT The) (JJ new) (NNS notes)) (VP (MD will) (VP (VB bear) (NP (NN interest)) (PP (PP (IN at) (NP (NP (CD 5.5) (NN %)) (PP (IN through) (NP (NP (NNP July) (CD 31)) (, ,) (NP (CD 1991)))))) (, ,) (CC and) (ADVP (RB thereafter)) (PP (IN at) (NP (CD 10) (NN %)))))) (. .)))", "(NP (NP (NNP July) (CD 31)) (, ,) (NP (CD 1991)))")
        # self.run_test(testPattern, "(ROOT (S (NP (NP (NNP Francis) (NNP M.) (NNP Wheat)) (, ,) (NP (NP (DT a) (JJ former) (NNPS Securities)) (CC and) (NP (NNP Exchange) (NNP Commission) (NN member))) (, ,)) (VP (VBD headed) (NP (NP (DT the) (NN panel)) (SBAR (WHNP (WDT that)) (S (VP (VBD had) (VP (VP (VBN studied) (NP (DT the) (NNS issues)) (PP (IN for) (NP (DT a) (NN year)))) (CC and) (VP (VBD proposed) (NP (DT the) (NNP FASB)) (PP (IN on) (NP (NP (NNP March) (CD 30)) (, ,) (NP (CD 1972))))))))))) (. .)))", "(NP (NP (NNP March) (CD 30)) (, ,) (NP (CD 1972)))")
        # self.run_test(testPattern, "(ROOT (S (NP (DT The) (NNP FASB)) (VP (VBD had) (NP (PRP$ its) (JJ initial) (NN meeting)) (PP (IN on) (NP (NP (NNP March) (CD 28)) (, ,) (NP (CD 1973))))) (. .)))", "(NP (NP (NNP March) (CD 28)) (, ,) (NP (CD 1973)))")
        # self.run_test(testPattern, "(ROOT (S (S (PP (IN On) (NP (NP (NNP Dec.) (CD 13)) (, ,) (NP (CD 1973)))) (, ,) (NP (PRP it)) (VP (VBD issued) (NP (PRP$ its) (JJ first) (NN rule)))) (: ;) (S (NP (PRP it)) (VP (VBD required) (S (NP (NNS companies)) (VP (TO to) (VP (VB disclose) (NP (NP (JJ foreign) (NN currency) (NNS translations)) (PP (IN in) (NP (NNP U.S.) (NNS dollars))))))))) (. .)))", "(NP (NP (NNP Dec.) (CD 13)) (, ,) (NP (CD 1973)))")
        # self.run_test(testPattern, "(ROOT (S (NP (NP (NNP Fidelity) (NNPS Investments)) (, ,) (NP (NP (DT the) (NN nation) (POS 's)) (JJS largest) (NN fund) (NN company)) (, ,)) (VP (VBD said) (SBAR (S (NP (NN phone) (NN volume)) (VP (VBD was) (NP (NP (QP (RBR more) (IN than) (JJ double)) (PRP$ its) (JJ typical) (NN level)) (, ,) (CC but) (ADVP (RB still)) (NP (NP (NN half) (DT that)) (PP (IN of) (NP (NP (NNP Oct.) (CD 19)) (, ,) (NP (CD 1987)))))))))) (. .)))", "(NP (NP (NNP Oct.) (CD 19)) (, ,) (NP (CD 1987)))")
        # self.run_test(testPattern, "(ROOT (S (NP (JJ SOFT) (NN CONTACT) (NNS LENSES)) (VP (VP (VBP WON) (NP (JJ federal) (NN blessing)) (PP (IN on) (NP (NP (NNP March) (CD 18)) (, ,) (NP (CD 1971))))) (, ,) (CC and) (VP (ADVP (RB quickly)) (VBD became) (NP (NN eye) (NNS openers)) (PP (IN for) (NP (PRP$ their) (NNS makers))))) (. .)))", "(NP (NP (NNP March) (CD 18)) (, ,) (NP (CD 1971)))")
        # self.run_test(testPattern, "(ROOT (NP (NP (NP (VBN Annualized) (NN interest) (NNS rates)) (PP (IN on) (NP (JJ certain) (NNS investments))) (SBAR (IN as) (S (VP (VBN reported) (PP (IN by) (NP (DT the) (NNP Federal) (NNP Reserve) (NNP Board))) (PP (IN on) (NP (DT a) (JJ weekly-average) (NN basis))))))) (: :) (NP-TMP (NP (CD 1989)) (CC and) (NP (NP (NNP Wednesday)) (NP (NP (NNP October) (CD 4)) (, ,) (NP (CD 1989))))) (. .)))", "(NP (NP (NNP October) (CD 4)) (, ,) (NP (CD 1989)))")
        # self.run_test(testPattern, "(ROOT (S (S (ADVP (RB Together))) (, ,) (NP (DT the) (CD two) (NNS stocks)) (VP (VP (VBD wreaked) (NP (NN havoc)) (PP (IN among) (NP (NN takeover) (NN stock) (NNS traders)))) (, ,) (CC and) (VP (VBD caused) (NP (NP (DT a) (ADJP (CD 7.3) (NN %)) (NN drop)) (PP (IN in) (NP (DT the) (NNP Dow) (NNP Jones) (NNP Transportation) (NNP Average))) (, ,) (ADJP (JJ second) (PP (IN in) (NP (NN size))) (PP (RB only) (TO to) (NP (NP (DT the) (NN stock-market) (NN crash)) (PP (IN of) (NP (NP (NNP Oct.) (CD 19)) (, ,) (NP (CD 1987)))))))))) (. .)))", "(NP (NP (NNP Oct.) (CD 19)) (, ,) (NP (CD 1987)))")

    def test_doc_examples(self):
        self.run_test("S < VP < NP", "(S (VP) (NP))", "(S (VP) (NP))")
        self.run_test("S < VP < NP", "(a (S (VP) (NP)) (S (NP) (VP)))", "(S (VP) (NP))", "(S (NP) (VP))")
        self.run_test("S < VP < NP", "(S (VP (NP)))")
        self.run_test("S < VP & < NP", "(S (VP) (NP))", "(S (VP) (NP))")
        self.run_test("S < VP & < NP", "(a (S (VP) (NP)) (S (NP) (VP)))", "(S (VP) (NP))", "(S (NP) (VP))")
        self.run_test("S < VP & < NP", "(S (VP (NP)))")
        self.run_test("S < VP << NP", "(S (VP (NP)))", "(S (VP (NP)))")
        self.run_test("S < VP << NP", "(S (VP) (foo NP))", "(S (VP) (foo NP))")
        self.run_test("S < (VP < NP)", "(S (VP (NP)))", "(S (VP (NP)))")
        self.run_test("S < (NP $++ VP)", "(S (NP) (VP))", "(S (NP) (VP))")
        self.run_test("S < (NP $++ VP)", "(S (NP VP))")

        self.run_test("(NP < NN || < NNS)", "((NP NN) (NP foo) (NP NNS))", "(NP NN)", "(NP NNS)")
        self.run_test(
            "(NP (< NN || < NNS) & > S)",
            "(foo (S (NP NN) (NP foo) (NP NNS)) (NP NNS))",
            "(NP NN)",
            "(NP NNS)",
        )
        self.run_test(
            "(NP [< NN || < NNS] & > S)",
            "(foo (S (NP NN) (NP foo) (NP NNS)) (NP NNS))",
            "(NP NN)",
            "(NP NNS)",
        )

    def test_complex(self):
        """
        More complex tests, often based on examples from our source code
        """
        test_pattern = "S < (NP=m1 $.. (VP < ((/VB/ < /^(am|are|is|was|were|'m|'re|'s|be)$/) $.. NP=m2)))"
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
            "(ROOT (S (NP (PRP$ My) (NN dog)) (VP (VBZ is) (VP (VBG eating) (NP (DT a) (NN sausage)))) (. .)))"
        )
        self.run_test(test_pattern, test_tree)

        test_tree = "(ROOT (S (NP (PRP He)) (VP (MD will) (VP (VB be) (ADVP (RB here) (RB soon)))) (. .)))"
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
        """
        More complex patterns to test
        """
        inputTrees = [
            (
                "(ROOT (S (NP (PRP You)) (VP (VBD did) (VP (VB go) (WHADVP (WRB How) (JJ long))"
                " (PP (IN for)))) (. .)))"
            ),
            (
                "(ROOT (S (NP (NNS Raccoons)) (VP (VBP do) (VP (VB come) (WHADVP (WRB When))"
                " (PRT (RP out)))) (. .)))"
            ),
            ("(ROOT (S (NP (PRP She)) (VP (VBZ is) (VP (WHADVP (WRB Where)) (VBG working))) (. .)))"),
            "(ROOT (S (NP (PRP You)) (VP (VBD did) (VP (WHNP (WP What)) (VB do))) (. .)))",
            (
                "(ROOT (S (NP (PRP You)) (VP (VBD did) (VP (VB do) (PP (IN in) (NP (NNP"
                " Australia))) (WHNP (WP What)))) (. .)))"
            ),
        ]

        pattern = "WHADVP=whadvp > VP $+ /[A-Z]*/=last ![$++ (PP < NP)]"
        self.run_test(pattern, inputTrees[0], "(WHADVP (WRB How) (JJ long))")
        self.run_test(pattern, inputTrees[1], "(WHADVP (WRB When))")
        self.run_test(pattern, inputTrees[2], "(WHADVP (WRB Where))")
        self.run_test(pattern, inputTrees[3])
        self.run_test(pattern, inputTrees[4])

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

    def test_link(self):
        # matched node will be (bar 3), the next bar matches (bar 2), and
        # the foo at the end obviously matches the (foo 1)
        self.run_test("bar $- (bar $- foo)", "(a (foo 1) (bar 2) (bar 3))", "(bar 3)")
        # same thing, but this tests the link functionality, as the
        # second match should also be (bar 2)
        self.run_test("bar=a $- (~a $- foo)", "(a (foo 1) (bar 2) (bar 3))", "(bar 3)")

        # TODO
        # won't work, since (bar 3) doesn't satisfy the next-to-foo
        # relation, and (bar 2) isn't the same node as (bar 3)
        # self.run_test("bar=a $- (=a $- foo)", "(a (foo 1) (bar 2) (bar 3))")

        # links can be saved as named nodes as well, so this should work
        self.run_test("bar=a $- (~a=b $- foo)", "(a (foo 1) (bar 2) (bar 3))", "(bar 3)")

        # run a few of the same tests, but this time dissect the results
        # to make sure the captured nodes are the correct nodes
        tree_string = "(a (foo 1) (bar 2) (bar 3))"
        pattern = TregexPattern("bar=a $- (~a $- foo)")
        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))
        self.assertEqual("(bar 3)", matches[0].tostring())
        self.assertEqual("(bar 3)", pattern.get_nodes("a")[0].tostring())

        pattern = TregexPattern("bar=a $- (~a=b $- foo)")
        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))
        self.assertEqual("(bar 3)", matches[0].tostring())
        self.assertEqual("(bar 3)", pattern.get_nodes("a")[0].tostring())
        self.assertEqual("(bar 2)", pattern.get_nodes("b")[0].tostring())

        pattern = TregexPattern("bar=a $- (~a=b $- foo=c)")
        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))
        self.assertEqual("(bar 3)", matches[0].tostring())
        self.assertEqual("(bar 3)", pattern.get_nodes("a")[0].tostring())
        self.assertEqual("(bar 2)", pattern.get_nodes("b")[0].tostring())
        self.assertEqual("(foo 1)", pattern.get_nodes("c")[0].tostring())

    # TODO
    #   def test_backref(self)
    # """Test another variant of using links, this time with pattern partitions"""
    #     TregexPattern tregex = TregexPattern.compile("__ <1 B=n <2 ~n");
    #     Tree tree = treeFromString("(A (B w) (B x))");
    #     TregexMatcher matcher = tregex.matcher(tree);
    #     assertTrue(matcher.find());
    #     Tree match = matcher.getMatch();
    #     assertEquals("(A (B w) (B x))", match.toString());
    #     Tree node = pattern.get_nodes("n");
    #     assertEquals("(B w)", node.toString());
    #     assertFalse(matcher.find());
    #
    #     tregex = TregexPattern.compile("__ < B=n <2 B=m : (=n !== =m)");
    #     tree = treeFromString("(A (B w) (B x))");
    #     matcher = tregex.matcher(tree);
    #     assertTrue(matcher.find());
    #     match = matcher.getMatch();
    #     assertEquals("(A (B w) (B x))", match.toString());
    #     node = pattern.get_nodes("n");
    #     assertEquals("(B w)", node.toString());
    #     assertFalse(matcher.find());

    def test_nonsense(self):
        # can't name a variable twice
        pattern = TregexPattern("foo=a $ bar=a")
        self.assertRaises(ParseException, pattern.findall, "(A)")

        # another way of doing the same thing
        pattern = TregexPattern("foo=a > bar=b $ ~a=b")
        self.assertRaises(ParseException, pattern.findall, "(A)")

        # ... but this should work
        TregexPattern("foo=a > bar=b $ ~a").findall("(A)")

        # can't link to a variable that doesn't exist yet
        pattern = TregexPattern("~a $- (bar=a $- foo)")
        self.assertRaises(ParseException, pattern.findall, "(A)")

        # can't reference a variable that doesn't exist yet
        # pattern = TregexPattern("=a $- (bar=a $- foo)")
        # self.assertRaises(ParseException, pattern.findall, '(A)')

        # you'd have to be really demented to do this
        pattern = TregexPattern("~a=a $- (bar=b $- foo)")
        self.assertRaises(ParseException, pattern.findall, "(A)")

        # This should work... no reason this would barf
        TregexPattern("foo=a : ~a").findall("(A)")
        TregexPattern("a < foo=a || < bar=a").findall("(A)")

        # TODO: pytregex indeed support this, remember to include this diverging behavior in pytregex's readme
        # can't have a link in one part of a disjunction to a variable in
        # another part of the disjunction; it won't be set if you get to
        # the ~a part, after all
        # TregexPattern("a < foo=a || < ~a").findall('(A)')

        # TODO: pytregex has not achieved the backreferencing feature
        # # same, but for references
        # try
        #   TregexPattern("a < foo=a | < =a")
        #   throw new RuntimeException("Expected a parse exception")
        # } catch (TregexParseException e)
        #   # yay, passed

        # can't name a variable under a negation
        pattern = TregexPattern("__ ! > __=a")
        self.assertRaises(ParseException, pattern.findall, "(A)")

        self.assertRaises(ParseException, self.run_test, "A=a < B=a < C=a", "")
        self.assertRaises(ParseException, self.run_test, "A=a < B=a", "")
        self.assertRaises(ParseException, self.run_test, "A=a [<B=a || <C=a]", "")
        self.assertRaises(ParseException, self.run_test, "A=a ?[<B=a || <C=a]", "")
        self.assertRaises(ParseException, self.run_test, "A=a ![<B=a || <C=a]", "")

    def test_numbered_sister(self):
        # this shouldn't mean anything
        pattern = TregexPattern("A $5 B")
        self.assertRaises(KeyError, pattern.findall, "(A)")

        # this should be fine
        TregexPattern("A <5 B").findall("(A)")

    def test_root_description(self):
        """test the _ROOT_ node description"""
        self.run_test("_ROOT_", "(ROOT (A apple))", "(ROOT (A apple))")
        self.run_test("A > _ROOT_", "(ROOT (A apple))", "(A apple)")
        self.run_test("A > _ROOT_", "(ROOT (A apple) (B (A aardvark)))", "(A apple)")
        self.run_test("A !> _ROOT_", "(ROOT (A apple) (B (A aardvark)))", "(A aardvark)")
        self.run_test("_ROOT_ <<<2 b", "(ROOT (A (B z) (C b)))", "(ROOT (A (B z) (C b)))")

    def test_only_match_root(self):
        tree_string = "(a (foo 1) (bar 2))"
        pattern = TregexPattern("__=a ! > __")
        matches = pattern.findall(tree_string)

        self.assertEqual(1, len(matches))
        self.assertEqual(tree_string, matches[0].tostring())
        self.assertEqual(tree_string, pattern.get_nodes("a")[0].tostring())

    def test_repeated_variables(self):
        tree_string = "(root (a (foo 1)) (a (bar 2)))"
        pattern = TregexPattern("a < foo=a || < bar=a")
        matches = pattern.findall(tree_string)

        self.assertEqual(2, len(matches))
        self.assertEqual("(a (foo 1))", matches[0].tostring())
        self.assertEqual("(foo 1)", pattern.get_nodes("a")[0].tostring())

        self.assertEqual("(a (bar 2))", matches[1].tostring())
        self.assertEqual("(bar 2)", pattern.get_nodes("a")[1].tostring())

    def test_more_curly_array(self):
        """
        A test case provided by a user which leverages variable names.
        Goal is to match this tree:
        (T
          (X
            (N
              (N Moe
                (PNT ,))))
          (NP
            (X
              (N Curly))
            (NP
              (CONJ and)
              (X
                (N Larry)))))
        """
        tree_string = "(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))"
        pattern = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, __=m <- (__ <, CONJ <- __=z))))")
        matches = pattern.findall(tree_string)

        self.assertEqual(1, len(matches))
        self.assertEqual("(PNT ,)", matches[0].tostring())
        self.assertEqual("(PNT ,)", pattern.get_nodes("p")[0].tostring())
        self.assertEqual("(X (N (N Moe (PNT ,))))", pattern.get_nodes("l")[0].tostring())
        self.assertEqual(tree_string, pattern.get_nodes("t")[0].tostring())
        self.assertEqual(
            "(NP (X (N Curly)) (NP (CONJ and) (X (N Larry))))",
            pattern.get_nodes("r")[0].tostring(),
        )
        self.assertEqual("(X (N Curly))", pattern.get_nodes("m")[0].tostring())
        self.assertEqual("(X (N Larry))", pattern.get_nodes("z")[0].tostring())

        # TODO: variable group
        # pattern = TregexPattern(
        #     "PNT=p >>- (/(.+)/#1%var=l >, (__=t <- (__=r <, /(.+)/#1%var=m <- (__ <, CONJ <-"
        #     " /(.+)/#1%var=z))))"
        # )
        # matches = pattern.findall(tree_string)
        # self.assertEqual(1, len(matches))
        # self.assertEqual("(PNT ,)", matches[0].tostring())
        # self.assertEqual("(PNT ,)", pattern.get_nodes("p")[0].tostring())
        # self.assertEqual("(X (N (N Moe (PNT ,))))", pattern.get_nodes("l")[0].tostring())
        # self.assertEqual(tree_string, pattern.get_nodes("t")[0].tostring())
        # self.assertEqual(
        #     "(NP (X (N Curly)) (NP (CONJ and) (X (N Larry))))",
        #     pattern.get_nodes("r")[0].tostring(),
        # )
        # self.assertEqual("(X (N Curly))", pattern.get_nodes("m")[0].tostring())
        # self.assertEqual("(X (N Larry))", pattern.get_nodes("z")[0].tostring())

        pattern = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, ~l <- (__ <, CONJ <- ~l))))")
        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))
        self.assertEqual("(PNT ,)", matches[0].tostring())
        self.assertEqual("(PNT ,)", pattern.get_nodes("p")[0].tostring())
        self.assertEqual("(X (N (N Moe (PNT ,))))", pattern.get_nodes("l")[0].tostring())
        self.assertEqual(tree_string, pattern.get_nodes("t")[0].tostring())
        self.assertEqual(
            "(NP (X (N Curly)) (NP (CONJ and) (X (N Larry))))",
            pattern.get_nodes("r")[0].tostring(),
        )

        pattern = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, ~l=m <- (__ <, CONJ <- ~l=z))))")
        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))
        self.assertEqual("(PNT ,)", matches[0].tostring())
        self.assertEqual("(PNT ,)", pattern.get_nodes("p")[0].tostring())
        self.assertEqual("(X (N (N Moe (PNT ,))))", pattern.get_nodes("l")[0].tostring())
        self.assertEqual(tree_string, pattern.get_nodes("t")[0].tostring())
        self.assertEqual(
            "(NP (X (N Curly)) (NP (CONJ and) (X (N Larry))))",
            pattern.get_nodes("r")[0].tostring(),
        )
        self.assertEqual("(X (N Curly))", pattern.get_nodes("m")[0].tostring())
        self.assertEqual("(X (N Larry))", pattern.get_nodes("z")[0].tostring())

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
        self.run_test("NP <# NNS", "(NP (NN work) (NNS practices))", "(NP (NN work) (NNS practices))")
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

    def test_immediate_sister(self):
        """
        Add a few more tests for immediate sister to make sure that $+
        doesn't accidentally match things that aren't non-immediate
        sisters, which should only be matched by $++
        """

        self.run_test(
            "@NP < (/^,/=comma $+ CC)",
            "((NP NP , NP , NP , CC NP))",
            "(NP NP , NP , NP , CC NP)",
        )
        self.run_test(
            "@NP < (/^,/=comma $++ CC)",
            "((NP NP , NP , NP , CC NP))",
            "(NP NP , NP , NP , CC NP)",
            "(NP NP , NP , NP , CC NP)",
            "(NP NP , NP , NP , CC NP)",
        )
        self.run_test(
            "@NP < (@/^,/=comma $+ @CC)",
            "((NP NP , NP , NP , CC NP))",
            "(NP NP , NP , NP , CC NP)",
        )

        pattern = TregexPattern("@NP < (/^,/=comma $+ CC)")

        tree_string = "(NP NP (, 1) NP (, 2) NP (, 3) CC NP)"
        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))
        self.assertEqual(tree_string, matches[0].tostring())
        node = pattern.get_nodes("comma")[0]
        self.assertEqual("(, 3)", node.tostring())

        tree_string = "(NP NP , NP , NP , CC NP)"
        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))
        self.assertEqual(tree_string, matches[0].tostring())

        node = pattern.get_nodes("comma")[0]
        self.assertEqual(",", node.tostring())

        self.assertEqual(5, node.get_sister_index())

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
        pattern = TregexPattern("/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ (< and) $+ (RB=adv $+ /^S/)))")
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(0, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(1, len(matches))

        # Third pattern: single relation in parentheses and negated.  Only first tree should match.
        pattern = TregexPattern("/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ !(< and) $+ (RB=adv $+ /^S/)))")
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(0, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(0, len(matches))

        # Fourth pattern: double relation in parentheses, no negation.
        pattern = TregexPattern("/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ (< and $+ RB) $+ (RB=adv $+ /^S/)))")
        matches = pattern.findall(tree_strings[0])
        self.assertEqual(0, len(matches))

        matches = pattern.findall(tree_strings[1])
        self.assertEqual(1, len(matches))

        matches = pattern.findall(tree_strings[2])
        self.assertEqual(1, len(matches))

        # Fifth pattern: double relation in parentheses, negated.
        pattern = TregexPattern("/^S/ < (/^S/ $++ (/^[,]|CC|CONJP$/ !(< and $+ RB) $+ (RB=adv $+ /^S/)))")
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
        self.run_test("A <= (A < B)", "(A (A (B 1)) (A (C 2)))", "(A (A (B 1)) (A (C 2)))", "(A (B 1))")
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
        self.run_test("A <... { B ; C ; D }", "(Z (A (B 1) (C 2) (D 3)))", "(A (B 1) (C 2) (D 3))")
        self.run_test("A <... { B ; C ; D }", "(A (B 1) (C 2) (D 3) (E 4))")
        self.run_test("A <... { B ; C ; D }", "(A (E 4) (B 1) (C 2) (D 3))")
        self.run_test("A <... { B ; C ; D }", "(A (B 1) (C 2) (E 4) (D 3))")
        self.run_test("A <... { B ; C ; D }", "(A (B 1) (C 2))")

        # every test above should return the opposite when negated
        self.run_test("A !<... { B ; C ; D }", "(A (B 1) (C 2) (D 3))")
        self.run_test("A !<... { B ; C ; D }", "(Z (A (B 1) (C 2) (D 3)))")
        self.run_test("A !<... { B ; C ; D }", "(A (B 1) (C 2) (D 3) (E 4))", "(A (B 1) (C 2) (D 3) (E 4))")
        self.run_test("A !<... { B ; C ; D }", "(A (E 4) (B 1) (C 2) (D 3))", "(A (E 4) (B 1) (C 2) (D 3))")
        self.run_test("A !<... { B ; C ; D }", "(A (B 1) (C 2) (E 4) (D 3))", "(A (B 1) (C 2) (E 4) (D 3))")
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
        tree_string = "(NP (UCP (NNP U.S.) (CC and) (ADJP (JJ northern) (JJ European))) (NNS diplomats))"
        pattern = TregexPattern("UCP [ <- (ADJP=adjp < JJR) || <, NNP=np ]")

        matches = pattern.findall(tree_string)
        self.assertEqual(1, len(matches))

        handled_nodes = pattern.get_nodes("np")
        self.assertIsNotNone(handled_nodes)
        self.assertEqual(1, len(handled_nodes))
        self.assertEqual("(NNP U.S.)", handled_nodes[0].tostring())

    def test_optional(self):
        tree_string = "(A (B (C 1)) (B 2))"
        pattern = TregexPattern("B ? < C=c")

        matches = pattern.findall(tree_string)
        self.assertEqual(2, len(matches))

        handled_nodes = pattern.get_nodes("c")
        self.assertIsNotNone(handled_nodes)
        self.assertEqual(1, len(handled_nodes))
        self.assertEqual("(C 1)", handled_nodes[0].tostring())

        tree_string = (
            "(ROOT (INTJ (CC But) (S (NP (DT the) (NNP RTC)) (ADVP (RB also)) (VP (VBZ requires)"
            " (`` ``) (S (FRAG (VBG working) ('' '') (NP (NP (NN capital)) (S (VP (TO to) (VP"
            " (VB maintain) (SBAR (S (NP (NP (DT the) (JJ bad) (NNS assets)) (PP (IN of) (NP (NP"
            " (NNS thrifts)) (SBAR (WHNP (WDT that)) (S (VP (VBP are) (VBN sold) (, ,) (PP (IN"
            " until) (NP (DT the) (NNS assets))))))))) (VP (MD can) (VP (VB be) (VP (VBN sold)"
            " (ADVP (RB separately))))))))))))))) (S (VP (. .)))))"
        )
        # TODO
        # # a pattern used to rearrange punctuation nodes in the srparser
        # pattern = TregexPattern(
        #     "__ !> __ <- (__=top <- (__ <<- (/[.]|PU/=punc < /[.!?。！？]/ ?> (__=single <:"
        #     " =punc))))"
        # )
        #
        # matches = pattern.findall(tree_string)
        # self.assertEqual(1, len(matches))
        #
        # self.assertEqual(1, len(pattern.get_nodes("punc")[0]))
        # self.assertEqual("(. .)", pattern.get_nodes("punc")[0].tostring())
        # # self.assertEqual("(VP (. .))", pattern.get_nodes("single")[0].tostring())

    def test_optional_subtree_pattern(self):
        # Tests the subtree pattern, <..., which checks for
        # an exact subtree under our current tree, but test it as an optional relation.
        #
        # Checks a bug reported by @tanloong (https://github.com/stanfordnlp/CoreNLP/issues/1375)
        self.run_test("A ?<... { C ; C ; D }", "(A (B 1) (C 2) (D 3))", "(A (B 1) (C 2) (D 3))")

    def test_optional_child(self):
        # The bug reported by @tanloong (https://github.com/stanfordnlp/CoreNLP/issues/1375)
        # actually applied to all optional coordination patterns
        #
        # Here we check that a simpler optional conjunction also fails
        self.run_test("A ?(< B <C)", "(A (B 1) (C 2) (D 3))", "(A (B 1) (C 2) (D 3))")

    def test_optional_child_miss(self):
        # An optional coordination which doesn't hit should also match exactly once
        self.run_test("A ?(< B < E)", "(A (B 1) (C 2) (D 3))", "(A (B 1) (C 2) (D 3))")

    def test_optional_disjunction(self):
        # An optional disjunction coordination should match at least once,
        # but not match any extra times just because of the optional
        # this matches once as an optional, even though none of the children match
        self.run_test("A ?[< E || < F]", "(A (B 1) (C 2) (D 3))", "(A (B 1) (C 2) (D 3))")

        # this matches twice
        self.run_test("A ?[< B || < C]", "(A (B 1) (C 2))", "(A (B 1) (C 2))", "(A (B 1) (C 2))")
        # this matches once, since the (< E) is useless
        self.run_test("A ?[< B || < E]", "(A (B 1) (C 2) (D 3))", "(A (B 1) (C 2) (D 3))")
        # now it will match twice, since the B should match twice
        self.run_test(
            "A ?[< B || < E]",
            "(A (B 1) (C 2) (B 3))",
            "(A (B 1) (C 2) (B 3))",
            "(A (B 1) (C 2) (B 3))",
        )

        # check by hand that foo & bar are set as expected for the disjunction matches
        # note that the order will be the order of the disjunction then subtrees,
        # not sorted by the order of the subtrees
        pattern = TregexPattern("A ?[< B=foo || < C=bar]")
        tree_string = "(A (B 1) (C 2) (B 3))"

        matches = pattern.findall(tree_string)
        self.assertEqual(3, len(matches))

        self.assertEqual("(B 1)", pattern.get_nodes("foo")[0].tostring())
        self.assertEqual("(B 3)", pattern.get_nodes("foo")[1].tostring())
        self.assertEqual("(C 2)", pattern.get_nodes("bar")[0].tostring())

        # this example should also work if the same name is used
        # for both of the children!
        pattern = TregexPattern("A ?[< B=foo || < C=foo]")
        matches = pattern.findall(tree_string)
        self.assertEqual(3, len(matches))

        self.assertEqual("(B 1)", pattern.get_nodes("foo")[0].tostring())
        self.assertEqual("(B 3)", pattern.get_nodes("foo")[1].tostring())
        self.assertEqual("(C 2)", pattern.get_nodes("foo")[2].tostring())

    def test_negated_disjunction(self):
        """
        A user supplied an example of a negated disjunction which went into an infinite loop.
        Apparently no one had ever used a negated disjunction of tree structures before!

        The problem was that the logic at the time tried to backtrack in
        the disjunction to find a better match, but that resulted in it
        going back and forth between the failed clause which was accepted
        and the successful clause which was rejected.  The problem being
        that the first half of the disjunction doesn't match, so the
        pattern is successful up to that point, but the second half does
        match, causing the pattern to be rejected and restarted.
        """
        self.run_test(
            "NP ![< /,/ || . (JJ<else)]",
            "( (NP (NP (NN anyone)) (ADJP (JJ else))))",
            "(NP (NP (NN anyone)) (ADJP (JJ else)))",
        )

    def test_disjunction_order(self):
        """
        Check output node order when patterns contain OR_REL, matchings that
        are indeed the same node should appear next to each other
        """
        self.run_test(
            "A [< B || < C]",
            "(A (B b1) (C c1)) (A (B b2))",
            "(A (B b1) (C c1))",
            "(A (B b1) (C c1))",
            "(A (B b2))",
        )

    def test_negated_exclusion(self):
        self.run_test("A !> A", "(A (A)) (A)", "(A (A))", "(A)")

    def test_normalize_escape(self):
        # normalize '-LRB-' to '(' when converting a treeString to a Tree
        # escape '(' to '-LRB-' when printing a Tree
        self.run_test(r"/\(/ < B", "(A (-LRB- B))", "(-LRB- B)")
        self.run_test(r"/\)/ < B", "(A (-RRB- B))", "(-RRB- B)")

    def run_test(self, pattern: Union[TregexPattern, str], tree_str: str, *expected_results: str):
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
            g = Tree.fromstring(expected_result)
            expected_tree = next(g, None)
            self.assertEqual(match.tostring(), expected_tree.tostring())
