#!/usr/bin/env python3

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Iterable, Iterator, List, NamedTuple, Optional

if TYPE_CHECKING:
    from pytregex.relation import AbstractRelationData
    from pytregex.tree import Tree


class NamedNodes:
    def __init__(self, name: Optional[str], nodes: Optional[List["Tree"]], string_repr: str = "") -> None:
        self.name = name
        self.nodes = nodes
        self.string_repr = string_repr

    def set_name(self, new_name: Optional[str]) -> None:
        self.name = new_name

    def set_nodes(self, new_nodes: List["Tree"]) -> None:
        self.nodes = new_nodes


class NodeDescription(NamedTuple):
    op: type["NODE_OP"]
    value: str

    def __repr__(self) -> str:
        return self.value


@dataclass
class BackRef:
    node_descriptions: "NodeDescriptions"
    nodes: Optional[list["Tree"]]


class NodeDescriptions:
    def __init__(
        self,
        *node_descriptions: NodeDescription,
        under_negation: bool = False,
        use_basic_cat: bool = False,
        condition: Optional["And"] = None,
        backref: Optional[BackRef] = None,
        name: Optional[str] = None,
    ) -> None:
        self.descriptions = list(node_descriptions)
        self.under_negation = under_negation
        self.use_basic_cat = use_basic_cat

        self.name = name
        self.condition = condition
        self.backref = backref

    def __iter__(self) -> Iterator[NodeDescription]:
        return iter(self.descriptions)

    def __repr__(self) -> str:
        if self.name is not None:
            ret = f"={self.name}"
        else:
            prefix = f"{'!' if self.under_negation else ''}{'@' if self.use_basic_cat else ''}"
            ret = f"{prefix}{'|'.join(map(str, self.descriptions))}"

        if self.condition is not None:
            ret = f"({ret} {self.condition})"
        return ret

    def set_backref(
        self,
        backref: BackRef,
        name: str,
    ) -> None:
        self.backref = backref
        self.name = name

    def set_condition(self, condition: "AbstractCondition") -> None:
        if self.condition is None:
            self.condition = And(condition)
        else:
            self.condition.append_condition(condition)

    def add_description(self, other_description: NodeDescription) -> None:
        self.descriptions.append(other_description)

    def negate(self) -> bool:
        if self.under_negation:
            return False

        self.under_negation = True
        return True

    def enable_basic_cat(self) -> bool:
        if self.use_basic_cat:
            return False

        self.use_basic_cat = True
        return True

    def _satisfies_ignore_condition(self, t: "Tree"):
        return any(
            desc.op.satisfies(
                t, desc.value, under_negation=self.under_negation, use_basic_cat=self.use_basic_cat
            )
            for desc in self.descriptions
        )

    def satisfies(self, t: "Tree") -> bool:
        if self.condition is None:
            return any(
                desc.op.satisfies(
                    t, desc.value, under_negation=self.under_negation, use_basic_cat=self.use_basic_cat
                )
                for desc in self.descriptions
            )
        else:
            cond_satisfies = self.condition.satisfies
            return any(
                desc.op.satisfies(
                    t, desc.value, under_negation=self.under_negation, use_basic_cat=self.use_basic_cat
                )
                and cond_satisfies(t)
                for desc in self.descriptions
            )

    def searchNodeIterator(self, t: "Tree", *, recursive: bool = True) -> Generator["Tree", None, None]:
        node_gen = t.preorder_iter() if recursive else (t for _ in range(1))
        node_gen = filter(self._satisfies_ignore_condition, node_gen)

        if self.condition is None:
            ret = node_gen
        else:
            cond_search = self.condition.searchNodeIterator
            ret = (m for node in node_gen for m in cond_search(node))

        if self.backref is not None:
            ret = list(ret)
            if self.backref.nodes is not None:
                self.backref.nodes.extend(ret)
            else:
                self.backref.nodes = ret
        yield from ret


class NODE_OP(ABC):
    @classmethod
    @abstractmethod
    def satisfies(
        cls,
        node: "Tree",
        value: str = "",
        *,
        under_negation: bool = False,
        use_basic_cat: bool = False,
    ) -> bool:
        raise NotImplementedError()

    @classmethod
    def in_(
        cls,
        node: "Tree",
        ids: Iterable[str],
        *,
        under_negation: bool = False,
        use_basic_cat: bool = False,
    ) -> bool:
        return any(
            cls.satisfies(node, id, under_negation=under_negation, use_basic_cat=use_basic_cat) for id in ids
        )


class NODE_ID(NODE_OP):
    @classmethod
    def satisfies(
        cls, node: "Tree", id: str, *, under_negation: bool = False, use_basic_cat: bool = False
    ) -> bool:
        attr = "basic_category" if use_basic_cat else "label"
        value = getattr(node, attr)

        if value is None:
            return under_negation
        else:
            return (value == id) != under_negation


class NODE_REGEX(NODE_OP):
    @classmethod
    def satisfies(
        cls, node: "Tree", regex: str, *, under_negation: bool = False, use_basic_cat: bool = False
    ) -> bool:
        attr = "basic_category" if use_basic_cat else "label"
        value = getattr(node, attr)

        if value is None:
            return under_negation
        else:
            # Convert regex to standard python regex
            flag = ""
            current_flag = regex[-1]
            while current_flag != "/":
                # Seems that only (?m) and (?x) are useful for node describing:
                #  re.ASCII      (?a)
                #  re.IGNORECASE (?i)
                #  re.LOCALE     (?L)
                #  re.DOTALL     (?s)
                #  re.MULTILINE  (?m)
                #  re.VERBOSE    (?x)
                if current_flag not in "xi":
                    raise ValueError(f"Error!! Unsupported regexp flag: {current_flag}")
                flag += current_flag
                regex = regex[:-1]
                current_flag = regex[-1]

            regex = regex[1:-1]
            if flag:
                regex = "(?" + "".join(set(flag)) + ")" + regex

            return (re.search(regex, value) is not None) != under_negation


class NODE_ANY(NODE_OP):
    @classmethod
    def satisfies(
        cls,
        node: "Tree",
        value: str = "",
        *,
        under_negation: bool = False,
        use_basic_cat: bool = False,
    ) -> bool:
        return not under_negation


class NODE_ROOT(NODE_OP):
    @classmethod
    def satisfies(
        cls,
        node: "Tree",
        value: str = "",
        *,
        under_negation: bool = False,
        use_basic_cat: bool = False,
    ) -> bool:
        return (node.parent is None) != under_negation


class AbstractCondition(ABC):
    # @abstractmethod
    # def match(self, these_nodes: List["Tree"], this_name: Optional[str]):
    #     pass
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def satisfies(self, t: "Tree") -> bool:
        try:
            next(self.searchNodeIterator(t))
        except StopIteration:
            return False
        else:
            return True

    @abstractmethod
    def searchNodeIterator(self, t: "Tree") -> Generator["Tree", None, None]:
        raise NotImplementedError


class Condition(AbstractCondition):
    def __init__(
        self,
        relation_data: "AbstractRelationData",
        node_descriptions: NodeDescriptions,
        # those_nodes: List["Tree"],
        # that_name: Optional[str],
    ) -> None:
        self.relation_data = relation_data
        self.node_descriptions = node_descriptions
        # self.satisfies = relation_data.satisfies
        # self.those_nodes = those_nodes
        # self.that_name = that_name

    def __repr__(self):
        return f"{self.relation_data} {self.node_descriptions}"

    def searchNodeIterator(self, t: "Tree") -> Generator["Tree", None, None]:
        for _ in self.relation_data.searchNodeIterator(t, self.node_descriptions):
            yield t

    # def get_names(self) -> Generator[Optional[str], None, None]:
    #     yield self.that_name
    #

    # def match(
    #     self, these_nodes: List["Tree"], this_name: Optional[str]
    # ) -> Tuple[List["Tree"], Dict[str, list]]:
    #     backrefs_map: Dict[str, list] = {}
    #
    #     matched_pairs = []
    #     for this_node in these_nodes:
    #         for that_node in self.those_nodes:
    #             if self.satisfies(this_node, that_node):
    #                 matched_pairs.append((this_node, that_node))
    #
    #     res = [pair[0] for pair in matched_pairs]
    #     if this_name is not None:
    #         backrefs_map[this_name] = res
    #
    #     if self.that_name is not None:
    #         that_name = self.that_name
    #         backrefs_map[that_name] = [pair[1] for pair in matched_pairs]
    #
    #     return res, backrefs_map


# ----------------------------------------------------------------------------
#                                   Logic


class And(AbstractCondition):
    def __init__(self, *conds: AbstractCondition):
        self.conditions = list(conds)

    def __repr__(self):
        return " ".join(map(str, self.conditions))

    def searchNodeIterator(self, t: "Tree") -> Generator["Tree", None, None]:
        candidates = (t,)
        for condition in self.conditions:
            candidates = tuple(
                node for candidate in candidates for node in condition.searchNodeIterator(candidate)
            )
        yield from candidates

    # def get_names(self) -> Generator[Optional[str], None, None]:
    #     for condition in self.conditions:
    #         for name in condition.get_names():
    #             yield name

    def append_condition(self, other_condition):
        self.conditions.append(other_condition)

    def extend_conditions(self, other_conditions):
        self.conditions.extend(other_conditions)

    # def match(
    #     self, these_nodes: List["Tree"], this_name: Optional[str]
    # ) -> Tuple[List["Tree"], Dict[str, list]]:
    #     backrefs_map: Dict[str, list] = {}
    #     old_these_nodes = these_nodes
    #
    #     for condition in self.conditions:
    #         these_nodes, backrefs_map_cur_cond = condition.match(these_nodes, this_name)
    #         backrefs_map.update(backrefs_map_cur_cond)
    #
    #         if not these_nodes:
    #             return [], {key: [] for key in backrefs_map}
    #
    #     # if some of these_nodes are filtered out, ensure that backrefs_map for names in every sub-conditions only satisfy the rest of these_nodes
    #     if len(these_nodes) < len(old_these_nodes) and any(name is not None for name in self.get_names()):
    #         for condition in self.conditions:
    #             _, backrefs_map_cur_cond = condition.match(these_nodes, this_name)
    #             backrefs_map.update(backrefs_map_cur_cond)
    #
    #     return these_nodes, backrefs_map


class Or(AbstractCondition):
    def __init__(self, *conds: AbstractCondition):
        self.conditions = list(conds)

    def __repr__(self):
        return " || ".join(map(str, self.conditions))

    def searchNodeIterator(self, t: "Tree") -> Generator["Tree", None, None]:
        for condition in self.conditions:
            yield from condition.searchNodeIterator(t)

    # def get_names(self) -> Generator[Optional[str], None, None]:
    #     for condition in self.conditions:
    #         for name in condition.get_names():
    #             yield name
    #
    def append_condition(self, other_condition):
        self.conditions.append(other_condition)

    def extend_conditions(self, other_conditions):
        self.conditions.extend(other_conditions)

    #
    # def match(
    #     self, these_nodes: List["Tree"], this_name: Optional[str]
    # ) -> Tuple[List["Tree"], Dict[str, list]]:
    #     backrefs_map: Dict[str, list] = {}
    #
    #     res = []
    #     for condition in self.conditions:
    #         res_cur_cond, backrefs_map_cur_cond = condition.match(these_nodes, this_name)
    #         for matched_this_node in res_cur_cond:
    #             for i, node in enumerate(res):
    #                 if matched_this_node is node:
    #                     res.insert(i, matched_this_node)
    #                     break
    #             else:
    #                 res.append(matched_this_node)
    #
    #         for name, nodes in backrefs_map_cur_cond.items():
    #             backrefs_map[name] = backrefs_map.get(name, []) + nodes
    #     return res, backrefs_map


class Not(AbstractCondition):
    def __init__(self, condition: AbstractCondition):
        self.condition = condition

    def __repr__(self):
        return f"!{self.condition}"

    def searchNodeIterator(self, t: "Tree") -> Generator["Tree", None, None]:
        try:
            next(self.condition.searchNodeIterator(t))
        except StopIteration:
            yield t
        else:
            return

    # def __init__(self, condition):
    #     self.condition = condition
    #
    #     for name in self.condition.get_names():
    #         if name is not None:
    #             raise SystemExit(
    #                 "Error!!  It is invalid to name a node that is under the scope of a negation"
    #                 f' operator. Please remove the assignment to "{name}".'
    #             )

    # def get_names(self) -> Generator[Optional[str], None, None]:
    #     for name in self.condition.get_names():
    #         yield name
    #
    # def match(
    #     self, these_nodes: List["Tree"], this_name: Optional[str]
    # ) -> Tuple[List["Tree"], Dict[str, list]]:
    #     matched_nodes, _ = self.condition.match(these_nodes, this_name)
    #     res = [
    #         this_node
    #         for this_node in these_nodes
    #         if all(this_node is not matched_node for matched_node in matched_nodes)
    #     ]
    #     return res, {}


class Opt(AbstractCondition):
    def __init__(self, condition: AbstractCondition):
        self.condition = condition

    def __repr__(self):
        return f"?[{self.condition}]"

    def searchNodeIterator(self, t: "Tree") -> Generator["Tree", None, None]:
        g = self.condition.searchNodeIterator(t)
        try:
            node = next(g)
        except StopIteration:
            yield t
        else:
            yield node
            yield from g

    # def get_names(self) -> Generator[Optional[str], None, None]:
    #     for name in self.condition.get_names():
    #         yield name
    #
    # def match(
    #     self, these_nodes: List["Tree"], this_name: Optional[str]
    # ) -> Tuple[List["Tree"], Dict[str, list]]:
    #     matched_nodes, backrefs_map = self.condition.match(these_nodes, this_name)
    #     if len(matched_nodes) >= len(these_nodes):
    #         these_nodes = matched_nodes
    #     return these_nodes, backrefs_map


"""
echo '(foo bar (rab (baz bar)))' | python -m pytregex 'foo=a <bar=a << baz=a' -filter -h a

echo '(foo bar (rab (baz bar)))' | python -m pytregex 'foo <bar=a << baz ' -filter -h a
echo '(foo bar (rab (baz bar)))' | tregex.py 'foo <bar=a << baz ' -filter -h a

echo '(foo bar (rab baz))' | python -m pytregex 'foo ![ <ba=z || << baz=r ]' -filter
echo '(foo bar (rab baz))' | tregex.py 'foo ![ <ba | << baz ]' -filter

echo '(foo bar (rab baz))' | python -m pytregex 'foo [ <bar || << baz ]' -filter
echo '(foo bar (rab baz))' | tregex.py 'foo [ <bar | << baz ]' -filter

echo '(foo )' | python -m pytregex 'foo=a $ bar=a' -filter

echo '(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))' | python -m pytregex 'PNT=p >>- (__=l >, (__=t <- (__=r <, __=m <- (__ <, CONJ <- __=z))))' -filter -h m
echo '(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))' | tregex.py 'PNT=p >>- (__=l >, (__=t <- (__=r <, __=m <- (__ <, CONJ <- __=z))))' -filter -h m

echo '(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))' | python -m pytregex '(__=r <, __=m <- (__ <, CONJ <- __=z))' -filter -h m
echo '(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))' | tregex.py '(__=r <, __=m <- (__ <, CONJ <- __=z))' -filter -h m

echo '(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))' | python -m pytregex '(__=r <- (__ <, CONJ <- __=z) <, __=m)' -filter -h m
echo '(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))' | tregex.py '(__=r <- (__ <, CONJ <- __=z) <, __=m)' -filter -h m


echo '(A (B 1) (C 2) (B 3))' | python -m pytregex 'A ?[< B=foo || < C=foo]' -filter
"""
