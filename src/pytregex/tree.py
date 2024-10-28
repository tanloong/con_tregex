import re
from collections import deque
from collections.abc import Generator, Iterator
from io import StringIO
from itertools import chain as _chain
from typing import Optional

from .peekable import peekable

# Reference: [CoreNLP](https://github.com/stanfordnlp/CoreNLP/blob/139893242878ecacde79b2ba1d0102b855526610/src/edu/stanford/nlp/trees/Tree.java)

LRB: str = "("
RRB: str = ")"
LRB_ESCAPE: str = "-LRB-"
RRB_ESCAPE: str = "-RRB-"
SPACE_SEPARATOR: str = " "


class Tree:
    TOKEN_RE = re.compile(rf"(?x) [{re.escape(LRB)}{re.escape(RRB)}] | [^\s{re.escape(LRB)}{re.escape(RRB)}]+")

    def __init__(
        self,
        label: str | None = None,
        children: list["Tree"] | None = None,
        parent: Optional["Tree"] = None,
    ):
        self.set_label(label)
        if children is None:
            self.children = []
        else:
            for child in children:
                child.set_parent(self)
            self.children = children
        # each subtree has at most one parent
        self.set_parent(parent)

    def set_label(self, label: str | None) -> None:
        if isinstance(label, str):
            self.label: str | None = self.normalize(label)
        elif label is None:
            self.label = None
        else:
            raise TypeError(f"label must be str, not {type(label).__name__}")

    def set_parent(self, node: Optional["Tree"]) -> None:
        self.parent = node

    def add_child(self, node: "Tree") -> None:
        node.set_parent(self)
        self.children.append(node)

    @classmethod
    def fromstring(cls, s: str) -> Generator["Tree", None, None]:
        """
        >>> t_gen = Tree.fromstring("( NP (DT The) (NN battery) (NN plant) )")
        >>> t = next(t_gen)
        >>> t.label
        'NP'
        >>> for node in t.children: print(node)
        (DT The)
        (NN battery)
        (NN plant)
        """
        stack_parent: deque[Tree] = deque()
        current_tree = None

        token_g = peekable(cls.TOKEN_RE.findall(s))
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
    def normalize(cls, s: str) -> str:
        return s.replace(RRB_ESCAPE, RRB).replace(LRB_ESCAPE, LRB)

    @classmethod
    def escape(cls, s: str) -> str:
        return s.replace(RRB, RRB_ESCAPE).replace(LRB, LRB_ESCAPE)

    @classmethod
    def _remove_extra_level(cls, root) -> "Tree":
        # get rid of extra levels of root with None label
        # e.g.: "((S (NP ...) (VP ...)))" -> "S (NP ...) (VP ...)"
        while root.label is None and len(root.children) == 1:
            root = root.children[0]
            root.parent = None
        return root

    def __repr__(self):
        with StringIO() as buf:
            stack: deque[Tree | str] = deque()
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

    def tostring(self) -> str:
        return repr(self)

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
        other_kids = other.children
        if len(my_kids) != len(other_kids):
            return False
        return all(my_kids[i] == other_kids[i] for i in range(len(my_kids)))

    def __hash__(self) -> int:
        # consider t1's hash different than t2's if they have different id, although t1==t2 might be True
        return id(self)

    def __bool__(self) -> bool:
        return not (self.label is None and not self.children)

    def is_leaf(self) -> bool:
        return not self.children

    def num_children(self) -> int:
        return len(self.children)

    def preorder_iter(self) -> Generator["Tree", None, None]:
        if not self:
            raise ValueError("Trying to iterate an empty tree")

        yield self
        iterator: Iterator = iter(self.children)
        while (node := next(iterator, None)) is not None:
            yield node
            if not node.is_leaf():
                iterator = _chain(node.children, iterator)

    def get_leaves(self) -> list["Tree"]:
        """
        Gets the leaves of the tree.  All leaves nodes are returned as a list
        ordered by the natural left to right order of the tree.  None values,
        if any, are inserted into the list like any other value.

        return a list of the leaves.
        """
        return [node for node in self.preorder_iter() if node.is_leaf()]

    def span_string(self) -> str:
        """
        Return String of leaves spanned by this tree

        >>> t_gen = Tree.fromstring("( NP (DT The) (NN battery) (NN plant) )")
        >>> t = next(t_gen)
        >>> t.span_string()
        'The battery plant'
        """
        return " ".join(leaf.tostring() for leaf in self.get_leaves() if leaf is not None)

    def _render(self, depth: Optional[int] = None, path: Optional[list] = None) -> list[str]:
        """
        Reference: https://github.com/astral-sh/uv/blob/6bc8639ce85075907aed67734c6d76539a72d319/crates/uv/src/commands/pip/tree.rs#L186

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
        """
        if path is None:
            path = []

        if depth is not None and len(path) >= depth:
            return []
        path.append(self.label)

        lines = [self.label if self.label is not None else ""]
        for idx, kid in enumerate(self.children):
            prefix_top, prefix_rest = ("└── ", "    ") if self.num_children() - 1 == idx else ("├── ", "│   ")
            lines.extend(
                f"{prefix_top if visited_idx == 0 else prefix_rest}{visited_line}"
                for visited_idx, visited_line in enumerate(kid._render(depth, path))
            )
        path.pop()
        return lines

    def render(self, depth: Optional[int] = None) -> str:
        """
        >>> t_gen = Tree.fromstring("( NP (DT The) (NN battery) (NN plant) )")
        >>> t = next(t_gen)
        >>> print(t.render())
        NP
        ├── DT
        │   └── The
        ├── NN
        │   └── battery
        └── NN
            └── plant
        >>> print(t.render(depth=2))
        NP
        ├── DT
        ├── NN
        └── NN
        """
        return "\n".join(self._render(depth=depth))
