#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from typing import List, Optional, TYPE_CHECKING

from collins_head_finder import CollinsHeadFinder

if TYPE_CHECKING:
    from head_finder import HeadFinder
    from tree import Tree

# translated from https://github.com/stanfordnlp/CoreNLP/blob/main/src/edu/stanford/nlp/trees/tregex/Relation.java
# last modified at Apr 3, 2022 (https://github.com/stanfordnlp/CoreNLP/commits/main/src/edu/stanford/nlp/trees/tregex/Relation.java)


class Relation:
    hf = CollinsHeadFinder()

    @classmethod
    def dominates(cls, t1: "Tree", t2: "Tree") -> bool:
        """
        `t1` and `t2` should be part of the same tree
        """
        while t2.parent is not None:
            if t2.parent is t1:
                return True
            else:
                t2 = t2.parent
        return False

    @classmethod
    def dominated_by(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.dominates(t2, t1)

    @classmethod
    def only_child_of(cls, t1: "Tree", t2: "Tree") -> bool:
        those_children = t2.children
        return len(those_children) == 1 and those_children[0] is t1

    @classmethod
    def has_only_child(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.only_child_of(t2, t1)

    @classmethod
    def last_child_of_parent(cls, t1: "Tree", t2: "Tree") -> bool:
        those_children = t2.children
        return len(those_children) > 0 and those_children[-1] is t1

    @classmethod
    def parent_of_last_child(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.last_child_of_parent(t2, t1)

    @classmethod
    def leftmost_child_of(cls, t1: "Tree", t2: "Tree") -> bool:
        those_children = t2.children
        return len(those_children) > 0 and those_children[0] is t1

    @classmethod
    def has_leftmost_child(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.leftmost_child_of(t2, t1)

    @classmethod
    def has_rightmost_descendant(cls, t1: "Tree", t2: "Tree") -> bool:
        if t1.is_leaf:
            return False
        last_child = t1.children[-1]
        return last_child is t2 or cls.has_rightmost_descendant(last_child, t2)

    @classmethod
    def rightmost_descendant_of(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.has_rightmost_descendant(t2, t1)

    @classmethod
    def has_leftmost_descendant(cls, t1: "Tree", t2: "Tree") -> bool:
        if t1.is_leaf:
            return False
        first_child = t1.children[0]
        return first_child is t2 or cls.has_leftmost_descendant(first_child, t2)

    @classmethod
    def leftmost_descendant_of(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.has_leftmost_descendant(t2, t1)

    @classmethod
    def left_sister_of(cls, t1: "Tree", t2: "Tree") -> bool:
        # t1 is t2 or t1 is root
        if t1 is t2 or t1.parent is None:
            return False

        sisters = t1.parent.children
        for i in range(len(sisters) - 1, 0, -1):  # from sisters[-1] to sisters[1]
            if sisters[i] is t1:
                return False
            if sisters[i] is t2:
                return True
        return False

    @classmethod
    def right_sister_of(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.left_sister_of(t2, t1)

    @classmethod
    def immediate_left_sister_of(cls, t1: "Tree", t2: "Tree") -> bool:
        # t1 is t2 or to is root
        if t1 is t2 or t1.parent is None:
            return False

        sisters = t1.parent.children
        for i in range(len(sisters) - 1, 0, -1):  # from sisters[-1] to sisters[1]
            if sisters[i] is t1:
                return False
            if sisters[i] is t2:
                return sisters[i - 1] is t1
        return False

    @classmethod
    def immediate_right_sister_of(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.immediate_left_sister_of(t2, t1)

    @classmethod
    def parent_of(cls, t1: "Tree", t2: "Tree") -> bool:
        return t2.parent is t1

    @classmethod
    def child_of(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.parent_of(t2, t1)

    @classmethod
    def sister_of(cls, t1: "Tree", t2: "Tree") -> bool:
        if t1 is t2 or t1.parent is None:
            return False
        parent = t1.parent
        return cls.parent_of(parent, t2)

    @classmethod
    def equals(cls, t1: "Tree", t2: "Tree") -> bool:
        return t1 is t2

    @classmethod
    def parent_equals(cls, t1: "Tree", t2: "Tree") -> bool:
        if t1 is t2:
            return True
        return cls.parent_of(t1, t2)

    @classmethod
    def unary_path_ancestor_of(cls, t1: "Tree", t2: "Tree") -> bool:
        if t1.is_leaf or t1.num_children > 1:
            return False
        only_child = t1.children[0]
        if only_child is t2:
            return True
        else:
            return cls.unary_path_ancestor_of(only_child, t2)

    @classmethod
    def unary_path_descedant_of(cls, t1: "Tree", t2: "Tree") -> bool:
        return cls.unary_path_ancestor_of(t2, t1)

    @classmethod
    def heads(cls, t1: "Tree", t2: "Tree", hf: Optional["HeadFinder"] = None) -> bool:
        if t2.is_leaf:
            return False
        elif t2.is_pre_terminal:
            return t2.first_child is t1
        else:
            if hf is None:
                hf = Relation.hf
            head = hf.determineHead(t2)
            if head is None:
                return False
            elif head is t1:
                return True
            else:
                return cls.heads(t1, head, hf)

    @classmethod
    def headed_by(cls, t1: "Tree", t2: "Tree", hf: Optional["HeadFinder"] = None) -> bool:
        return cls.heads(t2, t1, hf)

    @classmethod
    def immediately_heads(
        cls, t1: "Tree", t2: "Tree", hf: Optional["HeadFinder"] = None
    ) -> bool:
        if hf is None:
            hf = Relation.hf
        return hf.determineHead(t2) is t1

    @classmethod
    def immediately_headed_by(
        cls, t1: "Tree", t2: "Tree", hf: Optional["HeadFinder"] = None
    ) -> bool:
        return cls.immediately_heads(t2, t1, hf)

    @classmethod
    def precedes(cls, t1: "Tree", t2: "Tree") -> bool:
        return t1.right_edge <= t2.left_edge

    @classmethod
    def immediately_precedes(cls, t1: "Tree", t2: "Tree") -> bool:
        return t1.right_edge == t2.left_edge

    @classmethod
    def follows(cls, t1: "Tree", t2: "Tree") -> bool:
        return t2.right_edge <= t1.left_edge

    @classmethod
    def immediately_follows(cls, t1: "Tree", t2: "Tree") -> bool:
        return t2.right_edge == t1.left_edge

    @classmethod
    def ancestor_of_leaf(cls, t1: "Tree", t2: "Tree") -> bool:
        return t1 is not t2 and t2.is_leaf and Relation.dominates(t1, t2)

    @classmethod
    def unbroken_category_dominates(cls, t1: "Tree", t2: "Tree", arg: List["Tree"]) -> bool:
        def path_match_node(node: "Tree") -> bool:
            nonlocal arg
            return node in arg

        for kid in t1.children:
            if kid is t2:
                return True
            else:
                if path_match_node(kid) and Relation.unbroken_category_dominates(kid, t2, arg):
                    return True
        return False

    @classmethod
    def unbroken_category_is_dominated_by(cls, t1: "Tree", t2: "Tree", arg: List["Tree"]) -> bool:
        return Relation.unbroken_category_dominates(t2, t1, arg)

    @classmethod
    def unbroken_category_precedes(cls, t1: "Tree", t2: "Tree", arg: List["Tree"]) -> bool:
        def path_match_node(node: "Tree") -> bool:
            nonlocal arg
            return node in arg
        if t1.parent is None: # if t1 is root
            return False

        parent = t1.parent
        i = t1.get_sister_index() # if t1 is not root, i won't be None, no need to check whether i is None
        while i == (parent.num_children -1) and parent.parent is not None:
            t1 = parent
            parent = parent.parent
            i = t1.get_sister_index()

        if (i+1) < parent.num_children:
            following_node = parent.children[i+1]
        else:
            return False

        if following_node is t2:
            return True
        else:
            if path_match_node(following_node) and Relation.unbroken_category_precedes(following_node, t2, arg):
                return True
        return False

    @classmethod
    def unbroken_category_follows(cls, t1: "Tree", t2: "Tree", arg: List["Tree"]) -> bool:
        return Relation.unbroken_category_precedes(t2, t1, arg)
