# /home/tan/.local/share/stanford-tregex-2020-11-17/stanford-tregex-4.2.0-sources/edu/stanford/nlp/trees/tregex/Relation.java
from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from ply import lex, yacc

from relation import Relation
from tree import Tree


class NamedNodes:
    def __init__(self, name: Optional[str], nodes: List[Tree]):
        self.name = name
        self.nodes = nodes

    def merge(self, other: "NamedNodes"):
        self.name = other.name
        self.nodes.extend(other.nodes)


class RelationDataBase(ABC):
    def __init__(self, op: Callable, *, is_negated: bool = False, is_optional: bool = False):
        self.op = op
        self.is_negated = is_negated
        self.is_optional = is_optional

    @abstractmethod
    def condition_func(self, this_node: Tree, that_node: Tree):
        raise NotImplementedError()

    def toggle_negated(self):
        self.is_negated = not self.is_negated

    def toggle_optional(self):
        self.is_optional = not self.is_optional


class RelationData(RelationDataBase):
    def __init__(self, op: Callable, *, is_negated: bool = False, is_optional: bool = False):
        super().__init__(op, is_negated=is_negated, is_optional=is_optional)

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node)


class RelationWithStrArgData(RelationDataBase):
    def __init__(
        self,
        op: Callable,
        arg: List[Tree],
        *,
        is_negated: bool = False,
        is_optional: bool = False,
    ):
        super().__init__(op, is_negated=is_negated, is_optional=is_optional)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class RelationWithNumArgData(RelationDataBase):
    def __init__(
        self, op: Callable, arg: int, *, is_negated: bool = False, is_optional: bool = False
    ):
        super().__init__(op, is_negated=is_negated, is_optional=is_optional)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class MultiRelationData(RelationWithNumArgData):
    def __init__(
        self, op: Callable, arg: int, *, is_negated: bool = False, is_optional: bool = False
    ):
        super().__init__(op, arg, is_negated=is_negated, is_optional=is_optional)


class AndCondition:
    def __init__(self, *, relation_data: RelationDataBase, named_nodes: NamedNodes):
        for attr in ("condition_func", "is_negated", "is_optional"):
            setattr(self, attr, getattr(relation_data, attr))

        self.named_nodes = named_nodes

    def toggle_negated(self):
        self.is_negated = not self.is_negated


class NotAndCondition:
    def __init__(self, *, conditions: List[AndCondition]):
        self.conditions = conditions
        self.toggle_negated()

    def toggle_negated(self):
        for condition in self.conditions:
            condition.toggle_negated()


AND_CONDITIONS = List[Union[AndCondition, NotAndCondition]]


class TregexMatcherBase:  # {{{
    @classmethod
    def match_and(
        cls,
        this_node: Tree,
        this_name: Optional[str],
        and_condition: AndCondition,
    ) -> Tuple[int, dict]:
        condition_func, those, is_negated, is_optional = and_condition.condition_func, and_condition.named_nodes, and_condition.is_negated, and_condition.is_optional

        # is_negated and is_optional should not be both True
        if is_negated and is_optional:
            raise SystemExit("Error!!  Node cannot be both negated and optional.")

        # is_negated=False, is_optional=False
        if not is_negated and not is_optional:
            match_count, backrefs_map = cls._match_condition(
                this_node, this_name, those, condition_func
            )
        # is_negated=True, is_optional=False
        elif is_negated:
            match_count, backrefs_map = cls._match_negated_condition(
                this_node, those, condition_func
            )
        # is_negated=False, is_optional=True
        else:
            match_count, backrefs_map = cls._match_optional_condition(
                this_node, those, condition_func
            )
        return (match_count, backrefs_map)

    @classmethod
    def match_not_and(
        cls,
        this_node: Tree,
        this_name: Optional[str],
        not_and_condition: NotAndCondition,
    ) -> Tuple[int, dict]:
        conditions = not_and_condition.conditions
        match_count = 0
        backrefs_map: Dict[str, list] = {}

        for condition in conditions:
            if isinstance(condition, NotAndCondition):
                match_count_cur_cond, _ = cls.match_not_and(this_node, this_name, condition)
            else:
                those = condition.named_nodes
                that_name = those.name

                if that_name is not None:
                    raise SystemExit(
                        "Error!!  It is not allowed to name a node that is under the scope of a"
                        f' negation operator. You need to remove the "{that_name}" designation.'
                    )

                match_count_cur_cond, _ = cls.match_and(
                    this_node,
                    this_name,
                    condition,
                )

            if match_count_cur_cond > 0:
                match_count = 1
                if this_name is not None:
                    backrefs_map[this_name] = [this_node]
                break

        # match_count returned by not_and should be either 0 or 1
        return (match_count, backrefs_map)

    @classmethod
    def _match_condition(
        cls,
        this_node: Tree,
        this_name: Optional[str],
        those: NamedNodes,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[int, dict]:
        that_name, those_nodes = those.name, those.nodes

        backrefs_map: Dict[str, list] = {}

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
    def _match_negated_condition(
        cls,
        this_node: Tree,
        those: NamedNodes,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[int, dict]:
        those_nodes = those.nodes

        match_count = 0
        for that_node in those_nodes:
            if condition_func(this_node, that_node):
                return (0, {})
        return (1, {})

    @classmethod
    def _match_optional_condition(
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
        cls, trees: List[Tree], or_nodes: List[str], is_negated: bool = False
    ) -> Generator[Tree, Any, None]:
        def condition_func(candidate, or_nodes) -> bool:
            return (candidate in or_nodes) != is_negated

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
        cls, trees: List[Tree], regex: str, is_negated: bool = False
    ) -> Generator[Tree, Any, None]:
        pattern = re.compile(regex)
        for tree in trees:
            for node in tree.preorder_iter():
                if node.label is None:
                    if is_negated:
                        yield node
                    continue

                if (pattern.search(node.label) is None) == is_negated:
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

        for and_condition in and_conditions:
            if isinstance(and_condition, NotAndCondition):
                match_count_cur_cond, backrefs_map_cur_cond = cls.match_not_and(this_node, this_name, and_condition)
            else:
                that_name = and_condition.named_nodes.name
                if that_name is not None:
                    if and_condition.is_negated:
                        raise SystemExit(
                            "Error!!  It is not allowed to name a node that is under the scope of a"
                            f' negation operator. You need to remove the "{that_name}" designation.'
                        )
                    if that_name in names:
                        raise SystemExit(
                            f'Error!!  The name "{that_name}" has been assigned multiple times in a'
                            " single chain of and_conditions."
                        )
                    else:
                        names.append(that_name)

                match_count_cur_cond, backrefs_map_cur_cond = cls.match_and(
                    this_node,
                    this_name,
                    and_condition,
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
        or_conditions: List[AND_CONDITIONS],
    ) -> Tuple[List[Tree], dict]:
        res: List[Tree] = []
        backrefs_map: Dict[str, list] = {}
        for this_node in these_nodes:
            for and_conditions in or_conditions:
                match_count, backrefs_map_cur_conds = cls.and_(this_node, this_name, and_conditions)

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

    def get_nodes(self, name: str) -> Optional[List[Tree]]:
        return self.backrefs_map.get(name, None)

    def _reset_lexer_state(self):
        """
        reset lexer.lexpos to make the lexer reusable
        https://github.com/dabeaz/ply/blob/master/doc/ply.md#internal-lexer-state
        """
        self.lexer.lexpos = 0

    def make_parser(self, trees: List[Tree]):
        tokens = self.tokens

        precedence = (
            # keep consistency with Stanford Tregex
            # 1. "VP < NP < N" matches a VP which dominates both an NP and an N
            # 2. "VP < (NP < N)" matches a VP dominating an NP, which in turn dominates an N
            # ("left", "RELATION", "AND"),
            # https://github.com/dabeaz/ply/issues/215
            ("left", "IMAGINE_REDUCE"),
            ("left", "OR_REL"),
            ("left", "OR_NODE"),
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
            named_nodes : or_nodes %prec IMAGINE_REDUCE
            """
            logging.debug("following rule: named_nodes -> or_nodes")
            p[0] = NamedNodes(
                None,
                list(TregexMatcher.match_or_nodes(trees, p[1], is_negated=False)),
            )

        def p_not_or_nodes(p):
            """
            named_nodes : NOT or_nodes
            """
            logging.debug("following rule: named_nodes -> NOT or_nodes")
            p[0] = NamedNodes(
                None, list(TregexMatcher.match_or_nodes(trees, p[2], is_negated=True))
            )

        def p_regex(p):
            """
            named_nodes : REGEX
            """
            logging.debug("following rule: named_nodes -> REGEX")
            p[0] = NamedNodes(
                None,
                list(node for node in TregexMatcher.match_regex(trees, p[1], is_negated=False)),
            )

        def p_not_regex(p):
            """
            named_nodes : NOT REGEX
            """
            logging.debug("following rule: named_nodes -> NOT REGEX")
            p[0] = NamedNodes(
                None,
                list(node for node in TregexMatcher.match_regex(trees, p[2], is_negated=True)),
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

        def p_named_nodes_or_node_named_nodes(p):
            """
            named_nodes : named_nodes OR_NODE named_nodes
            """
            logging.debug("following rule: named_nodes -> named_nodes OR_NODE named_nodes")
            these, those = p[1], p[3]
            if these.name is not None:
                logging.critical(f"Error!!  It is not allowd to name a node within the node description. You need to remove the \"{these.name}\" designation or place it at the end of the node description.")
                raise SystemExit()
            these.merge(those)
            p[0] = these

        def p_lparen_named_nodes_rparen(p):
            """
            named_nodes : LPAREN named_nodes RPAREN
            """
            logging.debug("following rule: named_nodes -> LPAREN named_nodes RPAREN")
            p[0] = p[2]

        # 2. relation
        # 2.1 RELATION
        def p_relation(p):
            """
            relation_data : RELATION
            """
            logging.debug("following rule: relation_data -> RELATION")
            p[0] = RelationData(self.RELATION_MAP[p[1]])

        # 2.2 REL_W_STR_ARG
        def p_rel_w_str_arg_lparen_named_nodes_rparen(p):
            """
            relation_data : REL_W_STR_ARG LPAREN named_nodes RPAREN
            """
            logging.debug(
                "following rule: relation_data -> REL_W_STR_ARG LPAREN named_nodes RPAREN"
            )
            p[0] = RelationWithStrArgData(self.REL_W_STR_ARG_MAP[p[1]], p[3].nodes)

        # 2.3 REL_W_NUM_ARG
        def p_relation_number(p):
            """
            relation_data : RELATION NUMBER
            """
            logging.debug("following rule: relation_data -> RELATION NUMBER")
            rel, num = p[1:]
            if rel.endswith("-"):
                num = f"-{num}"
            p[0] = RelationWithNumArgData(self.REL_W_NUM_ARG_MAP[rel], int(num))

        def p_not_relation_data(p):
            """
            relation_data : NOT relation_data
            """
            logging.debug("following rule: relation_data -> NOT relation_data")
            p[2].toggle_negated()
            p[0] = p[2]

        def p_optional_relation_data(p):
            """
            relation_data : OPTIONAL relation_data
            """
            logging.debug("following rule: relation_data -> OPTIONAL relation_data")
            p[2].toggle_optional()
            p[0] = p[2]

        # 3. and_conditions
        def p_relation_data_named_nodes(p):
            """
            and_condition : relation_data named_nodes %prec IMAGINE_REDUCE
            """
            # %prec IMAGINE_REDUCE: https://github.com/dabeaz/ply/issues/215
            logging.debug("following rule: and_condition -> relation_data named_nodes")
            p[0] = AndCondition(relation_data=p[1], named_nodes=p[2])

        def p_and_and_condition(p):
            """
            and_condition : AND and_condition
            """
            logging.debug("following rule: and_condition -> AND and_condition")
            p[0] = p[2]

        def p_and_not_and_condition(p):
            """
            and_condition : AND not_and_condition
            """
            logging.debug("following rule: and_condition -> AND not_and_condition")
            p[0] = p[2]

        def p_and_condition(p):
            """
            and_conditions : and_condition
            """
            logging.debug("following rule: and_conditions -> and_condition")
            and_conditions = [p[1]]
            p[0] = and_conditions

        def p_not_and_condition(p):
            """
            and_conditions : not_and_condition
            """
            logging.debug("following rule: and_conditions -> not_and_condition")
            and_conditions = [p[1]]
            p[0] = and_conditions

        def p_and_conditions_and_condition(p):
            """
            and_conditions : and_conditions and_condition
            """
            logging.debug("following rule: and_conditions -> and_conditions and_condition")
            p[1].append(p[2])
            p[0] = p[1]

        def p_and_conditions_not_and_condition(p):
            """
            and_conditions : and_conditions not_and_condition
            """
            logging.debug("following rule: and_conditions -> and_conditions not_and_condition")
            p[1].append(p[2])
            p[0] = p[1]

        def p_multi_relation_named_nodes(p):
            """
            and_conditions_multi_relation : MULTI_RELATION "{" named_nodes_list "}"
            """
            logging.debug('following rule: and_conditions_multi_relation -> MULTI_RELATION "{" named_nodes_list "}"')
            op = self.MULTI_RELATION_MAP[p[1]]
            named_nodes_list = p[3]

            conditions = []

            for i, named_nodes in enumerate(named_nodes_list, 1):
                multi_relation_data = MultiRelationData(op, i)
                conditions.append(AndCondition(relation_data=multi_relation_data, named_nodes=named_nodes))

            any_named_nodes = NamedNodes(None, list(TregexMatcher.match_any(trees)))
            multi_relation_data = MultiRelationData(
                op, i + 1, is_negated=True  # type:ignore
            )
            conditions.append(AndCondition(relation_data = multi_relation_data, named_nodes=any_named_nodes))

            p[0] = conditions

        def p_and_conditions_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions and_conditions_multi_relation
            """
            logging.debug("following rule: and_conditions -> and_conditions and_conditions_multi_relation")
            p[1].extend(p[2])
            p[0] = p[1]

        def p_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions_multi_relation
            """
            logging.debug('following rule: and_conditions -> and_conditions_multi_relation')
            p[0] = p[1]

        def p_not_and_conditions_multi_relation(p):
            """
            not_and_condition : NOT and_conditions_multi_relation
            """
            logging.debug("following rule: not_and_condition -> NOT and_conditions_multi_relation")
            and_conditions = p[2]

            not_and_condition = NotAndCondition(conditions=and_conditions)

            p[0] = not_and_condition

        def p_lparen_and_conditions_rparen(p):
            """
            and_conditions : LPAREN and_conditions RPAREN
            """
            logging.debug("following rule: and_conditions : LPAREN and_conditions RPAREN")
            p[0] = p[2]

        def p_and_conditions(p):
            """
            or_conditions : and_conditions
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

        def p_not_lparen_or_conditions_rparen(p):
            """
            not_and_conditions : NOT LPAREN or_conditions RPAREN
            """
            logging.debug("following rule: not_and_conditions -> NOT LPAREN or_conditions RPAREN")
            or_conditions = p[3]
            not_and_conditions = []

            for and_conditions in or_conditions:
                not_and_condition = NotAndCondition(conditions=and_conditions)
                not_and_conditions.append(not_and_condition)

            p[0] = not_and_conditions

        def p_not_and_conditions(p):
            """
            and_conditions : not_and_conditions
            """
            logging.debug("following rule: and_conditions -> not_and_conditions")
            p[0] = p[1]

        def p_and_conditions_not_and_conditions(p):
            """
            and_conditions : and_conditions not_and_conditions
            """
            logging.debug("following rule: and_conditions -> and_conditions not_and_conditions")
            p[1].extend(p[2])
            p[0] = p[1]

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
            expr : named_nodes_list
            """
            logging.debug("following rule: expr -> named_nodes_list")
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

        return yacc.yacc(debug=True, start="expr")
