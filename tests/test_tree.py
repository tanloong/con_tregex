#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import re

from pytregex.tree import Tree

from .base_tmpl import BaseTmpl
from .base_tmpl import tree as tree_string


class TestTree(BaseTmpl):
    def setUp(self):
        self.tree_string = tree_string
        self.tree = next(Tree.fromstring(self.tree_string))
        return super().setUp()

    def test_fromstring(self):
        tree_string1 = "(NP (EX There)"
        tree_string2 = "NP (EX There))"
        tree_string3 = "(NP (EX There)))"

        g1 = Tree.fromstring(tree_string1)
        # File "pytregex/src/pytregex/tree.py", line 398, in fromstring
        #  raise ValueError("incomplete tree (extra left parentheses in input)")
        self.assertRaises(ValueError, next, g1)

        g2 = Tree.fromstring(tree_string2)
        next(g2)
        # File "pytregex/src/pytregex/tree.py", line 378, in fromstring
        #  ValueError: failed to build tree from string with extra non-matching right parentheses
        self.assertRaises(ValueError, next, g2)

        g3 = Tree.fromstring(tree_string3)
        next(g3)
        # File "pytregex/src/pytregex/tree.py", line 378, in fromstring
        #  ValueError: failed to build tree from string with extra non-matching right parentheses
        self.assertRaises(ValueError, next, g3)

        # make sure that extra levels of root with None label has been removed
        self.assertEqual(next(Tree.fromstring(f"(({tree_string}))")), next(Tree.fromstring(tree_string)))

    def test_set_label(self):
        tree = next(Tree.fromstring(self.tree_string))
        new_label = "TOOR"  # inverse of ROOT
        tree.set_label(new_label)
        self.assertEqual(tree.label, new_label)

        self.assertRaises(TypeError, tree.set_label, [new_label])

    def test_eq(self):
        from copy import deepcopy

        tree = next(Tree.fromstring(self.tree_string))

        # compare Tree with other classes
        self.assertNotEqual(tree, "non-Tree object")

        # compare trees with different labels
        tree2 = deepcopy(tree)
        self.assertEqual(tree, tree2)
        tree2.set_label("non-existing label")
        self.assertNotEqual(tree, tree2)

        # compare trees with the same label but different number of children
        tree2.set_label("ROOT")
        tree.set_label("ROOT")
        self.assertEqual(tree, tree2)
        tree2.children.append(Tree())
        self.assertNotEqual(tree, tree2)

        # compare trees with the same label, the same number of children, but different child label
        tree.children.append(Tree())
        self.assertEqual(tree, tree2)
        tree2.children[0].set_label("non-existing label for child")
        self.assertNotEqual(tree, tree2)

    def test_getitem(self):
        cases = [
            (self.tree[0], self.tree.children[0]),
            (self.tree[0, 0], self.tree.children[0].children[0]),
            (self.tree[0, 0], self.tree[0][0]),
            (self.tree[[]], self.tree),
            (self.tree[[0]], self.tree.children[0]),
            (self.tree[[0, 0]], self.tree.children[0].children[0]),
            (self.tree[[0, 0]], self.tree[0, 0]),
        ]
        for elem1, elem2 in cases:
            self.assertIs(elem1, elem2)

        self.assertRaises(TypeError, self.tree.__getitem__, "string")

    def test_len(self):
        tree = next(Tree.fromstring(self.tree_string))
        self.assertEqual(len(tree), len(tree.children))

    def test_num_children(self):
        tree = next(Tree.fromstring(self.tree_string))
        self.assertEqual(tree.num_children(), len(tree.children))

    def test_sister_index(self):
        tree = Tree()  # label=None, children=[], parent=None
        self.assertEqual(-1, tree.get_sister_index())

        self.assertEqual(self.tree[0].get_sister_index(), 0)
        self.assertEqual(self.tree[-1].get_sister_index(), len(self.tree.children) - 1)

        tree_S = Tree("S", children=[Tree("NP"), Tree("VP")])
        tree_VP = tree_S.children.pop()
        self.assertEqual(-1, tree_VP.get_sister_index())

    def test_is_leaf(self):
        tree = Tree()
        self.assertTrue(tree.is_leaf())

        tree.children.append(Tree())
        self.assertFalse(tree.is_leaf())

    def test_left_sisters(self):
        self.assertIsNone(self.tree[0].left_sisters, None)
        self.assertEqual(self.tree[0, 1, 1].left_sisters, self.tree[0, 1][:1])

    def test_right_sisters(self):
        self.assertIsNone(self.tree[0].right_sisters, None)
        self.assertEqual(self.tree[0, 1, 0].right_sisters, self.tree[0, 1][1:])

    def test_is_preterminal(self):
        tree = Tree()
        self.assertFalse(tree.is_pre_terminal)

        tree.children.append(Tree())
        self.assertTrue(tree.is_pre_terminal)

        tree.children.append(Tree())
        self.assertFalse(tree.is_pre_terminal)

    def test_tostring(self):
        tree = next(Tree.fromstring(self.tree_string))
        # suppress onto one line
        tree_string = re.sub(r"\n\s+", " ", self.tree_string.strip())
        self.assertEqual(tree.tostring(), tree_string)

    def test_root(self):
        child = self.tree[0]
        grandchild = self.tree[0, 0]
        self.assertIs(child.get_root(), self.tree)
        self.assertIs(grandchild.get_root(), self.tree)
