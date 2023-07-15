# /home/tan/.local/share/stanford-tregex-2020-11-17/stanford-tregex-4.2.0-sources/edu/stanford/nlp/trees/tregex/Relation.java
from dataclasses import dataclass
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from ply import lex, yacc
from relation import Relation
from tree import Tree


@dataclass
class NamedNodes:
    name: Optional[str]
    nodes: List[Tree]


class ReducedRelationBase(ABC):
    def __init__(self, op: Callable, modifier: Optional[str]):
        self.op = op
        self.modifier = modifier

    @abstractmethod
    def condition_func(self, this_node: Tree, that_node: Tree):
        raise NotImplementedError()


class ReducedRelation(ReducedRelationBase):
    def __init__(self, op: Callable, modifier: Optional[str]):
        super().__init__(op, modifier)

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node)


class ReducedRelationWithStrArg(ReducedRelationBase):
    def __init__(self, op: Callable, modifier: Optional[str], arg: List[Tree]):
        super().__init__(op, modifier)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class ReducedRelationWithNumArg(ReducedRelationBase):
    def __init__(self, op: Callable, modifier: Optional[str], arg: int):
        super().__init__(op, modifier)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class ReducedMultiRelation(ReducedRelationWithNumArg):
    def __init__(self, op: Callable, modifier: Optional[str], arg: int):
        super().__init__(op, modifier, arg)


MODIFIER = Optional[str]
AND_CONDITION = Tuple[ReducedRelationBase, NamedNodes]
AND_CONDITIONS = List[AND_CONDITION]


class TregexMatcherBase:  # {{{
    @classmethod
    def match_relation(
        cls,
        this_node: Tree,
        this_name: Optional[str],
        those: NamedNodes,
        modifier: MODIFIER,
        condition_func: Callable,
    ) -> Tuple[int, dict]:
        assert modifier in (None, "!", "?")
        if modifier is None:
            match_count, backrefs_map = cls._simply_match_relation(
                this_node, this_name, those, condition_func
            )
        elif modifier == "!":
            match_count, backrefs_map = cls._not_match_relation(this_node, those, condition_func)
        else:
            match_count, backrefs_map = cls._optionally_match_relation(
                this_node, those, condition_func
            )
        return (match_count, backrefs_map)

    @classmethod
    def _simply_match_relation(
        cls,
        this_node: Tree,
        this_name: Optional[str],
        those: NamedNodes,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[int, dict]:
        that_name, those_nodes = those.name, those.nodes

        backrefs_map: Dict[str, list] = {}
        # for "A=x < B=x", only map x to B
        if this_name is not None and this_name == that_name:
            this_name = None

        for name in (this_name, that_name):
            if name is not None:
                backrefs_map[name] = []

        match_count = 0
        for that_node in those_nodes:
            if condition_func(this_node, that_node):
                for name, node in ((this_name, this_node), (that_name, that_node)):
                    if name is not None:
                        backrefs_map[name].append(node)
                match_count += 1

        return (match_count, backrefs_map)

    @classmethod
    def _not_match_relation(
        cls,
        this_node: Tree,
        those: NamedNodes,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[int, dict]:
        those_nodes = those.nodes

        match_count = 0
        if all(
            map(
                lambda that_node, this_node=this_node: not condition_func(this_node, that_node),  # type: ignore
                those_nodes,
            )
        ):
            match_count += 1
        return (match_count, {})

    @classmethod
    def _optionally_match_relation(
        cls,
        this_node: Tree,
        those: NamedNodes,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[int, dict]:
        that_name, those_nodes = those.name, those.nodes

        if that_name is None:
            return (1, {})

        backrefs_map: Dict[str, list] = {}
        backrefs_map[that_name] = []
        for that_node in those_nodes:
            if condition_func(this_node, that_node):
                backrefs_map[that_name].append(that_node)
        return (1, backrefs_map)


# }}}
class TregexMatcher(TregexMatcherBase):
    @classmethod
    def match_any(cls, trees: List[Tree]) -> Generator[Tree, Any, None]:
        for tree in trees:
            for node in tree.preorder_iter():
                yield node

    @classmethod
    def match_or_nodes(
        cls, trees: List[Tree], or_nodes: List[str], is_negate: bool = False
    ) -> Generator[Tree, Any, None]:
        def condition_func(candidate, or_nodes) -> bool:
            return (candidate in or_nodes) != is_negate

        if or_nodes[0] == "@":
            attr = "basic_category"
            or_nodes.pop(0)
        else:
            attr = "label"

        for tree in trees:
            for node in tree.preorder_iter():
                if condition_func(getattr(node, attr), or_nodes):
                    yield node

    @classmethod
    def match_regex(
        cls, trees: List[Tree], regex: str, is_negate: bool = False
    ) -> Generator[Tree, Any, None]:
        pattern = re.compile(regex)
        for tree in trees:
            for node in tree.preorder_iter():
                if node.label is None:
                    if is_negate:
                        yield node
                    continue

                if (pattern.search(node.label) is None) == is_negate:
                    yield node

    @classmethod
    def and_(
        cls,
        this_node: Tree,
        this_name: Optional[str],
        and_conditions: AND_CONDITIONS,
    ) -> Tuple[int, dict]:
        match_count = 1
        backrefs_map: Dict[str, list] = {}

        # prevent the same name is given to different nodes
        names = [this_name] if this_name is not None else []
        for relation_data, those in and_conditions:
            that_name = those.name
            if that_name is not None:
                if that_name in names:
                    raise SystemExit(
                        f'Error!!  The name "{that_name}" has been assigned multiple times in a single'
                        " chain of and_conditions."
                    )
                else:
                    names.append(that_name)

            match_count_cur_cond, backrefs_map_cur_cond = cls.match_relation(
                this_node,
                this_name,
                those,
                relation_data.modifier,
                relation_data.condition_func,
            )

            if match_count_cur_cond == 0:
                return (0, {})

            match_count *= match_count_cur_cond
            for name, node_list in backrefs_map_cur_cond.items():
                backrefs_map[name] = node_list

        return (match_count, backrefs_map)

    @classmethod
    def or_(
        cls,
        these_nodes: List[Tree],
        this_name: Optional[str],
        or_conditions: Tuple[AND_CONDITIONS],
    ) -> Tuple[List[Tree], dict]:
        res: List[Tree] = []
        backrefs_map: Dict[str, list] = {}

        for this_node in these_nodes:
            for and_conditions in or_conditions:
                match_count, backrefs_map_cur_conds = cls.and_(
                    this_node, this_name, and_conditions
                )

                res += [this_node for _ in range(match_count)]
                for name, node_list in backrefs_map_cur_conds.items():
                    backrefs_map[name] = backrefs_map.get(name, []) + node_list
        return (res, backrefs_map)


class TregexPattern:
    RELATION_MAP = {
        "<": Relation.parent_of,
        ">": Relation.child_of,
        "<<": Relation.dominates,
        ">>": Relation.dominated_by,
        ">:": Relation.only_child_of,
        "<:": Relation.has_only_child,
        ">`": Relation.last_child_of_parent,
        ">-": Relation.last_child_of_parent,
        "<`": Relation.parent_of_last_child,
        "<-": Relation.parent_of_last_child,
        ">,": Relation.leftmost_child_of,
        "<,": Relation.has_leftmost_child,
        "<<`": Relation.has_rightmost_descendant,
        "<<-": Relation.has_rightmost_descendant,
        ">>`": Relation.rightmost_descendant_of,
        ">>-": Relation.rightmost_descendant_of,
        ">>,": Relation.leftmost_descendant_of,
        "<<,": Relation.has_leftmost_descendant,
        "$..": Relation.left_sister_of,
        "$++": Relation.left_sister_of,
        "$--": Relation.right_sister_of,
        "$,,": Relation.right_sister_of,
        "$.": Relation.immediate_left_sister_of,
        "$+": Relation.immediate_left_sister_of,
        "$-": Relation.immediate_right_sister_of,
        "$,": Relation.immediate_right_sister_of,
        "$": Relation.sister_of,
        "==": Relation.equals,
        "<=": Relation.parent_equals,
        "<<:": Relation.unary_path_ancestor_of,
        ">>:": Relation.unary_path_descedant_of,
        ":": Relation.pattern_splitter,
        ">#": Relation.immediately_heads,
        "<#": Relation.immediately_headed_by,
        ">>#": Relation.heads,
        "<<#": Relation.headed_by,
        "..": Relation.precedes,
        ",,": Relation.follows,
        ".": Relation.immediately_precedes,
        ",": Relation.immediately_follows,
        "<<<": Relation.ancestor_of_leaf,
        "<<<-": Relation.ancestor_of_leaf,
    }

    REL_W_STR_ARG_MAP = {
        "<+": Relation.unbroken_category_dominates,
        ">+": Relation.unbroken_category_is_dominated_by,
        ".+": Relation.unbroken_category_precedes,
        ",+": Relation.unbroken_category_follows,
    }

    REL_W_NUM_ARG_MAP = {
        ">": Relation.ith_child_of,
        ">-": Relation.ith_child_of,
        "<": Relation.has_ith_child,
        "<-": Relation.has_ith_child,
        "<<<": Relation.ancestor_of_ith_leaf,
        "<<<-": Relation.ancestor_of_ith_leaf,
    }

    MULTI_RELATION_MAP = {
        "<...": Relation.has_ith_child,
    }

    tokens = [
        "RELATION",
        "REL_W_STR_ARG",
        "MULTI_RELATION",
        "BLANK",
        "REGEX",
        "NOT",
        "OPTIONAL",
        "AND",
        "OR_NODE",
        "OR_REL",
        "LPAREN",
        "RPAREN",
        "LBRACKET",
        "RBRACKET",
        "EQUAL",
        "AT",
        "NUMBER",
        "ID",
        "TERMINATOR",
    ]

    # make sure long relations are checked first, or otherwise `>>` might
    # be tokenized as two `>`s.
    rels = sorted(RELATION_MAP.keys(), key=len, reverse=True)
    # add negative lookahead assertion to ensure ">+" is seen as REL_W_STR_ARG instead of RELATION(">") and ID("+")
    t_RELATION = r"(?:" + "|".join(map(re.escape, rels)) + r")(?![\+\.])"

    rels_w_arg = sorted(REL_W_STR_ARG_MAP.keys(), key=len, reverse=True)
    t_REL_W_STR_ARG = "|".join(map(re.escape, rels_w_arg))

    # REL_W_NUM_ARG don't have to be declared, as they have already been as t_RELATION

    multi_rels = sorted(MULTI_RELATION_MAP.keys(), key=len, reverse=True)
    t_MULTI_RELATION = "|".join(map(re.escape, multi_rels))

    t_BLANK = r"__"
    t_NOT = r"!"
    t_OPTIONAL = r"\?"
    t_AND = r"&"  # in `NP < NN | < NNS & > S`, `&` takes precedence over `|`
    t_OR_REL = r"\|\|"
    t_OR_NODE = r"\|"
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_LBRACKET = r"\["
    t_RBRACKET = r"\]"
    t_EQUAL = r"="
    t_AT = r"@"
    t_NUMBER = r"[0-9]+"
    t_ID = r"[^ 0-9\n\r(/|@!#&)=?[\]><~_.,$:{};][^ \n\r(/|@!#&)=?[\]><~.$:{};]*"
    t_TERMINATOR = r";"
    t_ignore = " \r\t"

    def t_REGEX(self, t):
        r"/[^/\n\r]*/[ix]*"
        flag = ""
        while t.value[-1] != "/":
            flag += t.value[-1]
            t.value = t.value[:-1]

        t.value = t.value[1:-1]
        if flag:
            t.value = "(?" + "".join(set(flag)) + ")" + t.value
        return t

    def t_error(self, t):
        logging.critical("Tokenization error: Illegal character '%s'" % t.value[0])
        raise SystemExit()

    literals = "{}"

    def __init__(self, tregex_pattern: str):
        self.lexer = lex.lex(module=self)
        self.lexer.input(tregex_pattern)

        self.backrefs_map: Dict[str, list] = {}
        self.pattern = tregex_pattern

    def findall(self, tree_string: str) -> List[Tree]:
        trees = Tree.from_string(tree_string)
        parser = self.make_parser(trees)
        self._reset_lexer_state()

        return parser.parse(lexer=self.lexer)

    def _reset_lexer_state(self):
        """
        reset lexer.lexpos to make the lexer reusable
        https://github.com/dabeaz/ply/blob/master/doc/ply.md#internal-lexer-state
        """
        self.lexer.lexpos = 0

    def make_parser(self, trees: List[Tree]):
        tokens = self.tokens

        precedence = (
            ("left", "OR_REL"),
            # keep consistency with Stanford Tregex
            # 1. "VP < NP < N" matches a VP which dominates both an NP and an N
            # 2. "VP < (NP < N)" matches a VP dominating an NP, which in turn dominates an N
            ("left", "RELATION", "AND"),
            # https://github.com/dabeaz/ply/issues/215
            ("left", "IMAGINE"),
            ("nonassoc", "EQUAL"),
        )

        # 1. Label description
        # 1.1 simple label description
        def p_id(p):
            """
            or_nodes : ID
            """
            logging.debug("following rule: or_nodes -> ID")
            p[0] = [p[1]]

        def p_or_or_nodes(p):
            """
            or_nodes : OR_NODE or_nodes
            """
            logging.debug("following rule: or_nodes -> OR_NODE or_nodes")
            p[0] = p[2]

        def p_or_nodes_or_nodes(p):
            """
            or_nodes : or_nodes or_nodes
            """
            logging.debug("following rule: or_nodes -> or_nodes or_nodes")
            p[1].extend(p[2])
            p[0] = p[1]

        def p_at_or_nodes(p):
            """
            or_nodes : AT or_nodes
            """
            logging.debug("following rule: or_nodes -> AT or_nodes")
            p[2].insert(0, p[1])
            p[0] = p[2]

        def p_lparen_or_nodes_rparen(p):
            """
            or_nodes : LPAREN or_nodes RPAREN
            """
            logging.debug("following rule: or_nodes -> LPAREN or_nodes RPAREN")
            p[0] = p[2]

        def p_or_nodes(p):
            """
            named_nodes : or_nodes
            """
            logging.debug("following rule: named_nodes -> or_nodes")
            p[0] = NamedNodes(
                None,
                list(TregexMatcher.match_or_nodes(trees, p[1], is_negate=False)),
            )

        def p_not_or_nodes(p):
            """
            named_nodes : NOT or_nodes
            """
            logging.debug("following rule: named_nodes -> NOT or_nodes")
            p[0] = NamedNodes(
                None, list(TregexMatcher.match_or_nodes(trees, p[2], is_negate=True))
            )

        def p_regex(p):
            """
            named_nodes : REGEX
            """
            logging.debug("following rule: named_nodes -> REGEX")
            p[0] = NamedNodes(
                None,
                list(node for node in TregexMatcher.match_regex(trees, p[1], is_negate=False)),
            )

        def p_not_regex(p):
            """
            named_nodes : NOT REGEX
            """
            logging.debug("following rule: named_nodes -> NOT REGEX")
            p[0] = NamedNodes(
                None,
                list(node for node in TregexMatcher.match_regex(trees, p[2], is_negate=True)),
            )

        def p_blank(p):
            """
            named_nodes : BLANK
            """
            logging.debug("following rule: named_nodes -> BLANK")
            p[0] = NamedNodes(None, list(TregexMatcher.match_any(trees)))

        def p_named_nodes_equal_id(p):
            """
            named_nodes : named_nodes EQUAL ID
            """
            logging.debug("following rule: named_nodes -> named_nodes EQUAL ID")
            name = p[3]
            nodes = p[1].nodes
            self.backrefs_map[name] = nodes

            p[0] = NamedNodes(name, nodes)

        def p_lparen_named_nodes_rparen(p):
            """
            named_nodes : LPAREN named_nodes RPAREN
            """
            logging.debug("following rule: named_nodes -> LPAREN named_nodes RPAREN")
            p[0] = p[2]

        # --------------------------------------------------------
        # 2. relation
        # 2.1 RELATION
        def p_relation(p):
            """
            reduced_relation : RELATION
            """
            logging.debug("following rule: reduced_relation -> RELATION")
            p[0] = ReducedRelation(self.RELATION_MAP[p[1]], None)

        def p_not_relation(p):
            """
            reduced_relation : NOT RELATION
            """
            logging.debug("following rule: reduced_relation -> NOT RELATION")
            p[0] = ReducedRelation(self.RELATION_MAP[p[2]], "!")

        def p_optional_relation(p):
            """
            reduced_relation : OPTIONAL RELATION
            """
            logging.debug("following rule: reduced_relation -> OPTIONAL RELATION")
            p[0] = ReducedRelation(self.RELATION_MAP[p[2]], "?")

        # 2.2 REL_W_STR_ARG
        def p_rel_w_str_arg_lparen_named_nodes_rparen(p):
            """
            reduced_rel_w_str_arg : REL_W_STR_ARG LPAREN named_nodes RPAREN
            """
            logging.debug(
                "following rule: reduced_rel_w_str_arg -> REL_W_STR_ARG LPAREN named_nodes"
                " RPAREN"
            )
            # relation=p[1], modifier=None, arg=p[3].nodes
            p[0] = ReducedRelationWithStrArg(self.REL_W_STR_ARG_MAP[p[1]], None, p[3].nodes)

        def p_not_rel_w_str_arg_lparen_named_nodes_rparen(p):
            """
            reduced_rel_w_str_arg : NOT REL_W_STR_ARG LPAREN named_nodes RPAREN
            """
            logging.debug(
                "following rule: reduced_rel_w_str_arg -> NOT REL_W_STR_ARG LPAREN named_nodes"
                " RPAREN"
            )
            # relation=p[2], modifier=None, arg=p[4].nodes
            p[0] = ReducedRelationWithStrArg(self.REL_W_STR_ARG_MAP[p[2]], "!", p[4].nodes)

        def p_optional_rel_w_str_arg_lparen_named_nodes_rparen(p):
            """
            reduced_rel_w_str_arg : OPTIONAL REL_W_STR_ARG LPAREN named_nodes RPAREN
            """
            logging.debug(
                "following rule: reduced_rel_w_str_arg -> OPTIONAL REL_W_STR_ARG LPAREN"
                " named_nodes RPAREN"
            )
            # relation=p[2], modifier=None, arg=p[4].nodes
            p[0] = ReducedRelationWithStrArg(self.REL_W_STR_ARG_MAP[p[2]], "?", p[4].nodes)

        # 2.3 REL_W_NUM_ARG
        def p_relation_number(p):
            """
            reduced_rel_w_num_arg : RELATION NUMBER
            """
            logging.debug("following rule: reduced_rel_w_num_arg -> RELATION NUMBER")
            rel, num = p[1:]
            if rel.endswith("-"):
                num = f"-{num}"
            p[0] = ReducedRelationWithNumArg(self.REL_W_NUM_ARG_MAP[rel], None, int(num))

        def p_not_relation_number(p):
            """
            reduced_rel_w_num_arg : NOT RELATION NUMBER
            """
            logging.debug("following rule: reduced_rel_w_num_arg -> NOT RELATION NUMBER")
            rel, num = p[2:]
            if rel.endswith("-"):
                num = f"-{num}"
            p[0] = ReducedRelationWithNumArg(self.REL_W_NUM_ARG_MAP[rel], "!", int(num))

        def p_optional_relation_number(p):
            """
            reduced_rel_w_num_arg : OPTIONAL RELATION NUMBER
            """
            rel, num = p[2:]
            if rel.endswith("-"):
                num = f"-{num}"
            logging.debug("following rule: reduced_rel_w_num_arg -> OPTIONAL RELATION NUMBER")
            p[0] = ReducedRelationWithNumArg(self.REL_W_NUM_ARG_MAP[rel], "?", int(num))

        # 3. and_conditions
        # --------------------------------------------------------
        def p_reduced_relation_named_nodes(p):
            """
            and_conditions : reduced_relation named_nodes %prec IMAGINE
            """
            logging.debug("following rule: and_conditions -> reduced_relation named_nodes")
            # %prec IMAGINE: https://github.com/dabeaz/ply/issues/215
            p[0] = [(p[1], p[2])]

        def p_multi_relation_named_nodes(p):
            """
            and_conditions : MULTI_RELATION "{" named_nodes_list "}"
            """
            logging.debug('following rule: and_conditions -> MULTI_RELATION "{" patterns "}"')
            multi_rel = p[1]
            named_nodes_list = p[3]

            op = self.MULTI_RELATION_MAP[multi_rel]
            and_conditions = []

            for i, named_nodes in enumerate(named_nodes_list, 1):
                reduced_multi_relation = ReducedMultiRelation(op, None, i)
                and_conditions.append((reduced_multi_relation, named_nodes))

            any_named_nodes = NamedNodes(None, list(TregexMatcher.match_any(trees)))
            reduced_multi_relation = ReducedMultiRelation(op, "!", i + 1)  # type:ignore
            and_conditions.append((reduced_multi_relation, any_named_nodes))

            p[0] = and_conditions

        def p_not_multi_relation_named_nodes(p):
            """
            or_conditions_multi_relation : NOT MULTI_RELATION "{" named_nodes_list "}"
            """
            logging.debug(
                'following rule: or_conditions_multi_relation -> NOT MULTI_RELATION "{"'
                ' patterns "}"'
            )
            multi_rel = p[2]
            named_nodes_list = p[4]

            op = self.MULTI_RELATION_MAP[multi_rel]
            or_conditions = []

            for i, named_nodes in enumerate(named_nodes_list, 1):
                # prevent naming subpattern children for NOT MULTI_RELATION (!<...)
                name = named_nodes.name
                if name is not None:
                    raise SystemExit(f"Error!!  Naming a subpattern for a negated MULTI_RELATION is not allowed. You need to remove the \"{name}\" designation.")
                
                reduced_multi_relation = ReducedMultiRelation(op, "!", i)
                or_conditions.append([(reduced_multi_relation, named_nodes)])

            any_named_nodes = NamedNodes(None, list(TregexMatcher.match_any(trees)))
            reduced_multi_relation = ReducedMultiRelation(op, None, i + 1)  # type:ignore
            or_conditions.append([(reduced_multi_relation, any_named_nodes)])

            p[0] = or_conditions

        def p_named_nodes_or_conditions_multi_relation(p):
            """
            named_nodes : named_nodes or_conditions_multi_relation
            """
            logging.debug(
                "following rule: named_nodes -> named_nodes or_conditions_multi_relation"
            )
            this_name, these_nodes = p[1].name, p[1].nodes
            or_conditions_multi_relation = p[2]
            res, backrefs_map = TregexMatcher.or_(
                these_nodes, this_name, or_conditions_multi_relation
            )
            for name, node_list in backrefs_map.items():
                logging.debug(
                    "Mapping {} to nodes:\n  {}".format(
                        name, "\n  ".join(node.to_string() for node in node_list)
                    )
                )
                self.backrefs_map[name] = node_list

            res_uniq = []
            previous_node: Optional[Tree] = None
            for node in res:
                if node is not previous_node:
                    res_uniq.append(node)
                previous_node = node

            p[0] = NamedNodes(this_name, res_uniq)

        def p_reduced_rel_w_str_arg_named_nodes(p):
            """
            and_conditions : reduced_rel_w_str_arg named_nodes %prec IMAGINE
            """
            logging.debug(
                "following rule: and_conditions -> reduced_rel_w_str_arg named_nodes %prec"
                " IMAGINE"
            )
            # %prec IMAGINE: https://github.com/dabeaz/ply/issues/215
            p[0] = [(p[1], p[2])]

        def p_reduced_rel_w_num_arg_named_nodes(p):
            """
            and_conditions : reduced_rel_w_num_arg named_nodes %prec IMAGINE
            """
            logging.debug(
                "following rule: and_conditions -> reduced_rel_w_num_arg named_nodes %prec"
                " IMAGINE"
            )
            # %prec IMAGINE: https://github.com/dabeaz/ply/issues/215
            p[0] = [(p[1], p[2])]

        def p_and_and_conditions(p):
            """
            and_conditions : AND and_conditions
            """
            logging.debug("following rule: and_conditions -> AND and_conditions")
            p[0] = p[2]

        # def p_or_conditions_and_conditions(p):
        #     """
        #     and_conditions : and_conditions and_conditions
        #     """
        #     logging.debug("following rule: and_conditions -> and_conditions and_conditions")
        #     p[1].extend(p[2])
        #     p[0] = p[1]

        # --------------------------------------------------------
        def p_and_condition(p):
            """
            or_conditions : and_conditions %prec IMAGINE
            """
            logging.debug("following rule: or_conditions -> and_conditions")
            p[0] = [p[1]]

        def p_or_conditions_or_or_conditions(p):
            """
            or_conditions : or_conditions OR_REL or_conditions
            """
            logging.debug("following rule: or_conditions -> or_conditions OR_REL or_conditions")
            p[1].extend(p[3])
            p[0] = p[1]

        def p_lparen_or_conditions_rparen(p):
            """
            or_conditions : LPAREN or_conditions RPAREN
            """
            logging.debug("following rule: or_conditions -> LPAREN or_conditions RPAREN")
            p[0] = p[2]

        def p_lbracket_or_conditions_rbracket(p):
            """
            or_conditions : LBRACKET or_conditions RBRACKET
            """
            logging.debug("following rule: or_conditions -> LBRACKET or_conditions RBRACKET")
            p[0] = p[2]

        def p_named_nodes_or_conditions(p):
            """
            named_nodes : named_nodes or_conditions
            """
            logging.debug("following rule: named_nodes -> named_nodes or_conditions")
            this_name, these_nodes = p[1].name, p[1].nodes
            or_conditions = p[2]
            res, backrefs_map = TregexMatcher.or_(these_nodes, this_name, or_conditions)
            for name, node_list in backrefs_map.items():
                logging.debug(
                    "Mapping {} to nodes:\n  {}".format(
                        name, "\n  ".join(node.to_string() for node in node_list)
                    )
                )
                self.backrefs_map[name] = node_list

            p[0] = NamedNodes(this_name, res)

        def p_named_nodes(p):
            """
            named_nodes_list : named_nodes
                               | named_nodes TERMINATOR
            """
            logging.debug("following rule: named_nodes_list -> named_nodes")
            # List[List[Tree]]
            p[0] = [p[1]]

        def p_named_nodes_list_named_nodes(p):
            """
            named_nodes_list : named_nodes_list named_nodes
                               | named_nodes_list named_nodes TERMINATOR
            """
            logging.debug("following rule: named_nodes_list -> named_nodes_list named_nodes")
            p[1].append(p[2])
            p[0] = p[1]

        def p_named_nodes_list(p):
            """
            pattern : named_nodes_list
            """
            logging.debug("following rule: pattern -> named_nodes_list")
            named_nodes_list = p[1]
            p[0] = list(node for named_nodes in named_nodes_list for node in named_nodes.nodes)

        def p_error(p):
            if p:
                logging.critical(
                    f"{self.lexer.lexdata}\n{' ' * p.lexpos}˄\nParsing error at token"
                    f" '{p.value}'"
                )
            else:
                logging.critical("Parsing Error at EOF")
            raise SystemExit()

        return yacc.yacc(debug=False, start="pattern")
