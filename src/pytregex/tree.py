# translated from [CoreNLP](https://github.com/stanfordnlp/CoreNLP/blob/139893242878ecacde79b2ba1d0102b855526610/src/edu/stanford/nlp/trees/Tree.java)

import re
from collections import deque
from collections.abc import Generator, Iterator
from io import StringIO
from itertools import chain as _chain
from typing import TYPE_CHECKING, Optional, Union

from .peekable import peekable

if TYPE_CHECKING:
    from .head_finder import HeadFinder

LRB: str = "("
RRB: str = ")"
LRB_ESCAPE: str = "-LRB-"
RRB_ESCAPE: str = "-RRB-"
SPACE_SEPARATOR: str = " "


class Tree:
    def __init__(
        self,
        label: Optional[str] = None,
        children: Optional[list["Tree"]] = None,
        parent: Optional["Tree"] = None,
    ):
        self.set_label(label)
        if children is None:
            self.children = []
        else:
            for child in children:
                child.parent = self  # type:ignore
            self.children = children
        # each subtree has at most one parent
        self.parent = parent

    def __repr__(self):
        # https://github.com/stanfordnlp/stanza/blob/c2d72bd14cf8cc28bd4e41a620692bbce5f43835/stanza/models/constituency/parse_tree.py#L118
        with StringIO() as buf:
            stack: deque[Union[Tree, str]] = deque()
            stack.append(self)
            while len(stack) > 0:
                node = stack.pop()
                if isinstance(node, str):
                    buf.write(node)
                    continue
                if len(node.children) == 0:
                    if node.label is not None:
                        buf.write(self.escape(node.label))
                    continue

                buf.write(LRB)
                if node.label is not None:
                    buf.write(self.escape(node.label))
                stack.append(RRB)

                for child in reversed(node.children):
                    stack.append(child)
                    stack.append(SPACE_SEPARATOR)
            buf.seek(0)
            return buf.read()

    def __eq__(self, other) -> bool:
        """
        Implements equality for Tree's.  Two Tree objects are equal if they
        have equal labels, the same number of children, and their children
        are pairwise equal. `t1 == t2` does not necessarily mean `t1 is t2`.

        :param other The object to compare with
        :return Whether two things are equal
        """
        if self.__class__ is not other.__class__:
            return False

        label1, label2 = self.label, other.label
        # if one or both of (self, other) has non-None label
        if (label1 is not None or label2 is not None) and (label1 is None or label1 != label2):
            return False

        my_kids = self.children
        their_kids = other.children
        if len(my_kids) != len(their_kids):
            return False
        return all(my_kids[i] == their_kids[i] for i in range(len(my_kids)))

    def __hash__(self) -> int:
        # consider t1's hash different than t2's if they have different id, although t1==t2 might be True
        return id(self)

    def __getitem__(self, index) -> "Tree":
        if isinstance(index, (int, slice)):
            return self.children[index]  # type:ignore
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                return self
            elif len(index) == 1:
                return self[index[0]]
            else:
                return self[index[0]][index[1:]]
        else:
            raise TypeError(f"{type(self).__name__} indices must be integers, not {type(index).__name__}")

    def __bool__(self) -> bool:
        return not (self.label is None and not self.children)

    def __len__(self) -> int:
        return len(self.children)

    @property
    def basic_category(self) -> Optional[str]:
        if self.label is None:
            return None
        return self.label.split("-")[0]

    def isLeaf(self) -> bool:
        return not self.children

    def numChildren(self) -> int:
        return len(self.children)

    def is_unary_rewrite(self) -> bool:
        """
        Says whether the current node has only one child. Can be used on an
        arbitrary Tree instance.

        return Whether the node heads a unary rewrite
        """
        return self.numChildren() == 1

    def is_preterminal(self) -> bool:
        """
        A preterminal is defined to be a node with one child which is itself a leaf.
        """
        return self.numChildren() == 1 and self.children[0].isLeaf()

    def is_prepreterminal(self) -> bool:
        """
        Return whether all the children of this node are preterminals or not.
        A preterminal is
        defined to be a node with one child which is itself a leaf.
        Considered false if the node has no children

        return true if the node is a prepreterminal; false otherwise
        """
        if self.numChildren() == 0:
            return False
        return all(child.is_preterminal() for child in self.children)

    def is_phrasal(self) -> bool:
        """
        Return whether this node is a phrasal node or not.  A phrasal node
        is defined to be a node which is not a leaf or a preterminal.
        Worded positively, this means that it must have two or more children,
        or one child that is not a leaf.

        return True if the node is phrasal; False otherwise
        """
        kids = self.children
        return not (kids is None or len(kids) == 0 or (len(kids) == 1 and kids[0].isLeaf()))

    def is_binary(self) -> bool:
        """
        Returns whether this node is the root of a possibly binary tree. This
        happens if the tree and all of its descendants are either nodes with
        exactly two children, or are preterminals or leaves.
        """
        if self.isLeaf() or self.is_preterminal():
            return True
        kids = self.children
        if len(kids) != 2:
            return False
        return kids[0].is_binary() and kids[1].is_binary()

    def firstChild(self) -> Optional["Tree"]:
        """
        Returns the first child of a tree, or None if none.

        return The first child
        """
        kids = self.children
        if len(kids) == 0:
            return None
        return kids[0]

    def lastChild(self) -> Optional["Tree"]:
        """
        Returns the last child of a tree, or None if none.

        return The last child
        """
        kids = self.children
        if len(kids) == 0:
            return None
        return kids[-1]

    def height(self) -> int:
        """
        The height of this tree.  The height of a tree containing no children
        is 1; the height of a tree containing only leaves is 2; and the height
        of any other tree is one plus the maximum of its children's heights.
        """
        if self.isLeaf():
            return 1

        stack, ret = [self], 0
        while stack:
            ret += 1
            tmp = []
            for node in stack:
                tmp.extend(node.children)
            stack = tmp
        return ret

    def head_terminal(self, hf: "HeadFinder") -> Optional["Tree"]:
        """
        Returns the tree leaf that is the head of the tree.

        param hf The head-finding algorithm to use
        param parent  The parent of this tree
        return The head tree leaf if any, else null
        """
        if self.isLeaf():
            return self

        head: Optional[Tree] = hf.determineHead(self)
        if head is not None:
            return head.head_terminal(hf)

        return None

    def get_terminal_labels(self) -> list[Optional[str]]:
        """
        Gets labels of terminal nodes. The Label of all leaf nodes is returned
        as a list ordered by the natural left to right order of the leaves.
        Null values, if any, are inserted into the list like any other value.

        return a List of the data in the tree's leaves.
        """
        return [leaf.label for leaf in self.getLeaves()]

    def get_tagged_terminal_labels(self, divider: str = "/") -> list[str]:
        """
        Gets the tagged labels of the tree -- that is, get the preterminals as
        well as the terminals.  The Label of all leaf nodes is returned as a
        list ordered by the natural left to right order of the leaves.  Null
        values, if any, are inserted into the list like any other value. This
        has been rewritten to thread, so only one List is used.

        param ty The list in which the tagged yield of the tree will be placed.
        Normally, this will be empty when the routine is called, but if not,
        the new yield is added to the end of the list.
        return a List of the data in the tree's leaves.
        """
        ret = []
        for node in self.preorder_iter():
            if node.is_preterminal():
                ret.append(f"{node.firstChild().label}{divider}{node.label}")  # type: ignore
        return ret

    def leftEdge(self) -> int:
        """
        note: return 0 for the leftmost node
        """
        i = 0

        def left_edge_helper(t: "Tree", t1: "Tree") -> bool:
            nonlocal i
            if t is t1:
                return True
            elif t1.isLeaf():
                j = len(t1.get_terminal_labels())
                i += j
                return False
            else:
                return any(left_edge_helper(t, kid) for kid in t1.children)

        if left_edge_helper(self, self.getRoot()):
            return i
        else:
            raise RuntimeError("Tree is not a descendant of root.")

    def rightEdge(self) -> int:
        """
        note: return 1 for the leftmost node
        """
        i = len(self.getRoot().get_terminal_labels())

        def right_edge_helper(t: "Tree", t1: "Tree") -> bool:
            nonlocal i
            if t is t1:
                return True
            elif t1.isLeaf():
                j = len(t1.get_terminal_labels())
                i -= j
                return False
            else:
                return any(right_edge_helper(t, kid) for kid in reversed(t1.children))

        if right_edge_helper(self, self.getRoot()):
            return i
        else:
            raise RuntimeError("Tree is not a descendant of root.")

    def get_sister_index(self) -> int:
        """Return -1 for root"""
        if self.parent is None:
            return -1
        for i, child in enumerate(self.parent.children):
            if child is self:
                return i
        return -1

    def set_label(self, label: Optional[str]) -> None:
        if isinstance(label, str):
            self.label: Optional[str] = self.normalize(label)
        elif label is None:
            self.label = None
        else:
            raise TypeError(f"label must be str, not {type(label).__name__}")

    def set_parent(self, node: "Tree") -> None:
        self.parent = node

    def add_child(self, node: "Tree") -> None:
        node.set_parent(self)
        self.children.append(node)

    @classmethod
    def normalize(cls, s: str) -> str:
        return s.replace(RRB_ESCAPE, RRB).replace(LRB_ESCAPE, LRB)

    @classmethod
    def escape(cls, s: str) -> str:
        return s.replace(RRB, RRB_ESCAPE).replace(LRB, LRB_ESCAPE)

    @classmethod
    def fromstring(cls, s: str) -> Generator["Tree", None, None]:
        # TODO need more logging msg to indicate whether "a b c d" or "(a b c d)" is parsed correctly
        # translated from CoreNLP's PennTreeReader
        # https://github.com/stanfordnlp/CoreNLP/blob/main/src/edu/stanford/nlp/trees/PennTreeReader.java#L144

        open_pattern = re.escape(LRB)
        close_pattern = re.escape(RRB)

        # store `token_re` to avoid repeated regex compiling
        attr = "token_re"
        if (token_re := getattr(cls, attr, None)) is None:
            token_re = re.compile(
                rf"(?x) [{open_pattern}{close_pattern}] | [^\s{open_pattern}{close_pattern}]+"
            )
            setattr(cls, attr, token_re)

        stack_parent: deque[Tree] = deque()
        current_tree = None

        token_g = peekable(token_re.findall(s))
        while (token := next(token_g, None)) is not None:
            if token == LRB:
                label = None if token_g.peek() == LRB else next(token_g, None)

                if label == RRB:
                    continue

                new_tree = cls(label)

                if current_tree is None:
                    stack_parent.append(new_tree)
                else:
                    current_tree.add_child(new_tree)
                    stack_parent.append(current_tree)

                current_tree = new_tree
            elif token == RRB:
                if len(stack_parent) == 0:
                    raise ValueError(
                        "failed to build tree from string with extra non-matching right parentheses"
                    )
                else:
                    current_tree = stack_parent.pop()

                    if len(stack_parent) == 0:
                        yield cls._remove_extra_level(current_tree)

                        current_tree = None
                        continue
            else:
                if current_tree is None:
                    continue

                new_tree = cls(token)
                current_tree.add_child(new_tree)

        if current_tree is not None:
            raise ValueError("incomplete tree (extra left parentheses in input)")

    @classmethod
    def _remove_extra_level(cls, root) -> "Tree":
        # get rid of extra levels of root with None label
        # e.g.: "((S (NP ...) (VP ...)))" -> "S (NP ...) (VP ...)"
        while root.label is None and len(root.children) == 1:
            root = root.children[0]
            root.parent = None
        return root

    def getRoot(self) -> "Tree":
        root_ = self
        while root_.parent is not None:
            root_ = root_.parent
        return root_

    def left_sisters(self) -> Optional[list]:
        sister_index_ = self.get_sister_index()
        is_not_the_first = sister_index_ is not None and sister_index_ > 0
        if is_not_the_first:
            return self.parent.children[:sister_index_]  # type:ignore
        return None

    def right_sisters(self) -> Optional[list]:
        sister_index_ = self.get_sister_index()
        is_not_the_last = sister_index_ is not None and sister_index_ < (
            len(self.parent.children) - 1  # type:ignore
        )
        if is_not_the_last:
            return self.parent.children[sister_index_ + 1 :]  # type:ignore
        return None

    def tostring(self) -> str:
        return repr(self)

    def preorder_iter(self) -> Generator["Tree", None, None]:
        if not self:
            raise ValueError("Trying to iterate an empty tree")

        yield self
        iterator: Iterator = iter(self.children)
        while (node := next(iterator, None)) is not None:
            yield node
            if not node.isLeaf():
                iterator = _chain(node.children, iterator)

    def getLeaves(self) -> list["Tree"]:
        """
        Gets the leaves of the tree.  All leaves nodes are returned as a list
        ordered by the natural left to right order of the tree.  None values,
        if any, are inserted into the list like any other value.

        return a list of the leaves.
        """
        return [node for node in self.preorder_iter() if node.isLeaf()]

    def span_string(self) -> str:
        """
        Return String of leaves spanned by this tree
        """
        return " ".join(leaf.tostring() for leaf in self.getLeaves() if leaf is not None)

    def _render(self, depth: Optional[int] = None, path: Optional[list] = None) -> list[str]:
        """
        For sub-visited nodes, add the prefix to make the tree display user-friendly.
        The key observation here is you can group the tree as follows when you're at the
        root of the tree:
        root_node
        ├── level_1_0          // Group 1
        │   ├── level_2_0      ...
        │   │   ├── level_3_0  ...
        │   │   └── level_3_1  ...
        │   └── level_2_1      ...
        ├── level_1_1          // Group 2
        │   ├── level_2_2      ...
        │   └── level_2_3      ...
        └── level_1_2          // Group 3
            └── level_2_4      ...

        The lines in Group 1 and 2 have `├── ` at the top and `|   ` at the rest while
        those in Group 3 have `└── ` at the top and `    ` at the rest.
        This observation is true recursively even when looking at the subtree rooted
        at `level_1_0`.

        Reference: https://github.com/astral-sh/uv/blob/6bc8639ce85075907aed67734c6d76539a72d319/crates/uv/src/commands/pip/tree.rs#L186
        """
        if path is None:
            path = []

        if depth is not None and len(path) >= depth:
            return []
        path.append(self.label)

        lines = [self.label if self.label is not None else ""]
        for idx, kid in enumerate(self.children):
            prefix_top, prefix_rest = ("└── ", "    ") if self.numChildren() - 1 == idx else ("├── ", "│   ")
            lines.extend(
                f"{prefix_top if visited_idx == 0 else prefix_rest}{visited_line}"
                for visited_idx, visited_line in enumerate(kid._render(depth, path))
            )
        path.pop()
        return lines

    def render(self, depth: Optional[int] = None) -> str:
        return "\n".join(self._render(depth=depth))
