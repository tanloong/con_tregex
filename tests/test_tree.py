#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import re

from con_tregex.tree import Tree

from .base_tmpl import BaseTmpl
from .base_tmpl import tree as tree_string


class TestTree(BaseTmpl):
    def setUp(self):
        self.tree_string = tree_string
        self.tree = Tree.from_string(self.tree_string)[0]
        return super().setUp()

    def test_tokenize(self):
        tokens = [match.group() for match in Tree._tokenize(self.tree_string)]
        # fmt: off
        tree_list = ['(', 'ROOT', '(', 'S', '(', 'NP', '(', 'EX', 'There', ')', ')', '(', 'VP', '(', 'VBD', 'was',
                     ')', '(', 'NP', '(', 'NP', '(', 'DT', 'no', ')', '(', 'NN', 'possibility', ')', ')', '(',
                     'PP', '(', 'IN', 'of', ')', '(', 'S', '(', 'VP', '(', 'VBG', 'taking', ')', '(', 'NP', '(',
                     'DT', 'a', ')', '(', 'NN', 'walk', ')', ')', '(', 'NP', '(', 'DT', 'that', ')', '(', 'NN',
                     'day', ')', ')', ')', ')', ')', ')', ')', '(', '.', '.', ')', ')', ')']
        # fmt: on
        self.assertEqual(tokens, tree_list)

    def test_from_string(self):
        tree_string = "(ROOT (S (NP (EX There))))"
        # a tree with unpaired parenthesis is illegal
        self.assertRaises(ValueError, Tree.from_string, tree_string.rstrip(")"))
        self.assertRaises(ValueError, Tree.from_string, tree_string.lstrip("("))

        # see whether extra levels of root with None label has been removed
        self.assertEqual(Tree.from_string(f"(({tree_string}))"), Tree.from_string(tree_string))

    def test_set_label(self):
        tree = Tree.from_string(self.tree_string)[0]
        new_label = "TOOR"  # inverse of ROOT
        tree.set_label(new_label)
        self.assertEqual(tree.label, new_label)

        self.assertRaises(TypeError, tree.set_label, [new_label])

    def test_eq(self):
        from copy import deepcopy

        tree = Tree.from_string(self.tree_string)[0]

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
        tree = Tree.from_string(self.tree_string)[0]
        self.assertEqual(len(tree), len(tree.children))

    def test_num_children(self):
        tree = Tree.from_string(self.tree_string)[0]
        self.assertEqual(tree.num_children, len(tree.children))

    def test_sister_index(self):
        tree = Tree()  # label=None, children=[], parent=None
        self.assertIsNone(tree.get_sister_index())

        self.assertEqual(self.tree[0].get_sister_index(), 0)
        self.assertEqual(self.tree[-1].get_sister_index(), len(self.tree.children) - 1)

        tree_S = Tree("S", children=[Tree("NP"), Tree("VP")])
        tree_VP = tree_S.children.pop()
        self.assertRaises(ValueError, tree_VP.get_sister_index)

    def test_is_leaf(self):
        tree = Tree()
        self.assertTrue(tree.is_leaf)

        tree.children.append(Tree())
        self.assertFalse(tree.is_leaf)

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

    def test_to_string(self):
        tree = Tree.from_string(self.tree_string)[0]
        # suppress onto one line
        tree_string = re.sub(r"\n\s+", " ", self.tree_string.strip())
        self.assertEqual(tree.to_string(), tree_string)

    def test_root(self):
        child = self.tree[0]
        grandchild = self.tree[0, 0]
        self.assertIs(child.root, self.tree)
        self.assertIs(grandchild.root, self.tree)
