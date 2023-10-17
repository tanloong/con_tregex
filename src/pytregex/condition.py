#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from tree import Tree
    from relation import AbstractRelationData


class Condition(ABC):
    @abstractmethod
    def match(self, these_nodes: List["Tree"], this_name: Optional[str]):
        pass


class ConditionOp(Condition):
    def __init__(
        self,
        relation_data: "AbstractRelationData",
        those_nodes: List["Tree"],
        that_name: Optional[str],
    ) -> None:
        self.condition_func = relation_data.condition_func
        self.those_nodes = those_nodes
        self.that_name = that_name

    def get_names(self) -> Generator[Optional[str], Any, None]:
        yield self.that_name

    def match(
        self, these_nodes: List["Tree"], this_name: Optional[str]
    ) -> Tuple[List["Tree"], Dict[str, list]]:
        backrefs_map: Dict[str, list] = {}

        matched_pairs = []
        for this_node in these_nodes:
            for that_node in self.those_nodes:
                if self.condition_func(this_node, that_node):
                    matched_pairs.append((this_node, that_node))

        res = [pair[0] for pair in matched_pairs]
        if this_name is not None:
            backrefs_map[this_name] = res

        if self.that_name is not None:
            that_name = self.that_name
            backrefs_map[that_name] = [pair[1] for pair in matched_pairs]

        return res, backrefs_map


# ----------------------------------------------------------------------------
#                                   Logic


class And(Condition):
    def __init__(self, conditions):
        self.conditions = conditions

    def get_names(self) -> Generator[Optional[str], Any, None]:
        for condition in self.conditions:
            for name in condition.get_names():
                yield name

    def append_condition(self, other_condition):
        self.conditions.append(other_condition)

    def extend_conditions(self, other_conditions):
        self.conditions.extend(other_conditions)

    def match(
        self, these_nodes: List["Tree"], this_name: Optional[str]
    ) -> Tuple[List["Tree"], Dict[str, list]]:
        backrefs_map: Dict[str, list] = {}
        old_these_nodes = these_nodes

        for condition in self.conditions:
            these_nodes, backrefs_map_cur_cond = condition.match(these_nodes, this_name)
            backrefs_map.update(backrefs_map_cur_cond)

            if not these_nodes:
                return [], {key: [] for key in backrefs_map}

        # if some of these_nodes are filtered out, ensure that backrefs_map for names in every sub-conditions only satisfy the rest of these_nodes
        if len(these_nodes) < len(old_these_nodes) and any(
            name is not None for name in self.get_names()
        ):
            for condition in self.conditions:
                _, backrefs_map_cur_cond = condition.match(these_nodes, this_name)
                backrefs_map.update(backrefs_map_cur_cond)

        return these_nodes, backrefs_map


class Or(Condition):
    def __init__(self, conditions):
        self.conditions = conditions

    def get_names(self) -> Generator[Optional[str], Any, None]:
        for condition in self.conditions:
            for name in condition.get_names():
                yield name

    def append_condition(self, other_condition):
        self.conditions.append(other_condition)

    def extend_conditions(self, other_conditions):
        self.conditions.extend(other_conditions)

    def match(
        self, these_nodes: List["Tree"], this_name: Optional[str]
    ) -> Tuple[List["Tree"], Dict[str, list]]:
        backrefs_map: Dict[str, list] = {}

        res = []
        for condition in self.conditions:
            res_cur_cond, backrefs_map_cur_cond = condition.match(these_nodes, this_name)
            for matched_this_node in res_cur_cond:
                for i, node in enumerate(res):
                    if matched_this_node is node:
                        res.insert(i, matched_this_node)
                        break
                else:
                    res.append(matched_this_node)

            for name, nodes in backrefs_map_cur_cond.items():
                backrefs_map[name] = backrefs_map.get(name, []) + nodes
        return res, backrefs_map


class Not(Condition):
    def __init__(self, condition):
        self.condition = condition

        for name in self.condition.get_names():
            if name is not None:
                raise SystemExit(
                    "Error!!  It is invalid to name a node that is under the scope of a negation"
                    f' operator. Please remove the assignment to "{name}".'
                )

    def get_names(self) -> Generator[Optional[str], Any, None]:
        for name in self.condition.get_names():
            yield name

    def match(
        self, these_nodes: List["Tree"], this_name: Optional[str]
    ) -> Tuple[List["Tree"], Dict[str, list]]:
        matched_nodes, _ = self.condition.match(these_nodes, this_name)
        res = [
            this_node
            for this_node in these_nodes
            if all(this_node is not matched_node for matched_node in matched_nodes)
        ]
        return res, {}


class Opt(Condition):
    def __init__(self, condition):
        self.condition = condition

    def get_names(self) -> Generator[Optional[str], Any, None]:
        for name in self.condition.get_names():
            yield name

    def match(
        self, these_nodes: List["Tree"], this_name: Optional[str]
    ) -> Tuple[List["Tree"], Dict[str, list]]:
        matched_nodes, backrefs_map = self.condition.match(these_nodes, this_name)
        if len(matched_nodes) >= len(these_nodes):
            these_nodes = matched_nodes
        return these_nodes, backrefs_map


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
