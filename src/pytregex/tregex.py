# /home/tan/.local/share/stanford-tregex-2020-11-17/stanford-tregex-4.2.0-sources/edu/stanford/nlp/trees/tregex/Relation.java
from abc import ABC, abstractmethod
from collections import namedtuple
import logging
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, Iterator

from ply import lex, yacc

from relation import Relation
from tree import Tree


class NamedNodes:
    def __init__(self, name: Optional[str], nodes: List[Tree], describing_str: str = ""):
        self.name = name
        self.nodes = nodes
        self.describing_str = describing_str

    def set_name(self, new_name: Optional[str]):
        self.name = new_name

    def set_nodes(self, new_nodes: List[Tree]):
        self.nodes = new_nodes

    def merge(self, other: "NamedNodes"):
        self.name = other.name
        self.nodes.extend(other.nodes)


class RelationDataBase(ABC):
    def __init__(
        self, rel_key: str, op: Callable, *, is_negated: bool = False, is_optional: bool = False
    ):
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
    def __init__(
        self, rel_key: str, op: Callable, *, is_negated: bool = False, is_optional: bool = False
    ):
        super().__init__(rel_key, op, is_negated=is_negated, is_optional=is_optional)

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node)


class RelationWithStrArgData(RelationDataBase):
    def __init__(
        self,
        rel_key: str,
        op: Callable,
        *,
        arg: List[Tree],
        is_negated: bool = False,
        is_optional: bool = False,
    ):
        super().__init__(rel_key, op, is_negated=is_negated, is_optional=is_optional)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class RelationWithNumArgData(RelationDataBase):
    def __init__(
        self,
        rel_key: str,
        op: Callable,
        *,
        arg: int,
        is_negated: bool = False,
        is_optional: bool = False,
    ):
        super().__init__(rel_key, op, is_negated=is_negated, is_optional=is_optional)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class MultiRelationData(RelationWithNumArgData):
    def __init__(
        self,
        rel_key: str,
        op: Callable,
        *,
        arg: int,
        is_negated: bool = False,
        is_optional: bool = False,
    ):
        super().__init__(rel_key, op, arg=arg, is_negated=is_negated, is_optional=is_optional)


class AndCondition:
    def __init__(self, *, relation_data: RelationDataBase, named_nodes: NamedNodes):
        self.condition_func, self.is_negated, self.is_optional = (
            relation_data.condition_func,
            relation_data.is_negated,
            relation_data.is_optional,
        )

        self.named_nodes = named_nodes

    def toggle_negated(self):
        self.is_negated = not self.is_negated


class NotAndCondition:
    def __init__(self, *, conditions: List[AndCondition]):
        self.conditions = conditions
        for condition in self.conditions:
            condition.toggle_negated()


AND_CONDITIONS = List[Union[AndCondition, NotAndCondition]]


class TregexMatcherBase:  # {{{
    @classmethod
    def match_id(
        cls, node: Tree, id: str, *, is_negated: bool = False, use_basic_cat: bool = False
    ) -> bool:
        attr = "basic_category" if use_basic_cat else "label"
        value = getattr(node, attr)

        if value is None:
            return is_negated
        else:
            return (value == id) != is_negated

    @classmethod
    def match_regex(
        cls, node: Tree, regex: str, *, is_negated: bool = False, use_basic_cat: bool = False
    ) -> bool:
        attr = "basic_category" if use_basic_cat else "label"
        value = getattr(node, attr)

        if value is None:
            return is_negated
        else:
            # convert regex to standard python regex
            flag = ""
            while regex[-1] != "/":
                flag += regex[-1]
                regex = regex[:-1]

            regex = regex[1:-1]
            if flag:
                regex = "(?" + "".join(set(flag)) + ")" + regex

            return (re.search(regex, value) is not None) != is_negated

    @classmethod
    def match_blank(
        cls,
        node: Tree,
        value: Optional[str] = None,
        *,
        is_negated: bool = False,
        use_basic_cat: bool = False,
    ) -> bool:
        return not is_negated

    @classmethod
    def match_and(
        cls, this_node: Tree, this_name: Optional[str], and_condition: AndCondition
    ) -> Tuple[int, dict]:
        condition_func, those, is_negated, is_optional = (
            and_condition.condition_func,
            and_condition.named_nodes,
            and_condition.is_negated,
            and_condition.is_optional,
        )

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
        return match_count, backrefs_map

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
        return match_count, backrefs_map

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

        return match_count, backrefs_map

    @classmethod
    def _match_negated_condition(
        cls,
        this_node: Tree,
        those: NamedNodes,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[int, dict]:
        those_nodes = those.nodes

        for that_node in those_nodes:
            if condition_func(this_node, that_node):
                return 0, {}
        return 1, {}

    @classmethod
    def _match_optional_condition(
        cls,
        this_node: Tree,
        those: NamedNodes,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[int, dict]:
        that_name, those_nodes = those.name, those.nodes

        if that_name is None:
            return 1, {}

        backrefs_map: Dict[str, list] = {}
        backrefs_map[that_name] = []
        for that_node in those_nodes:
            if condition_func(this_node, that_node):
                backrefs_map[that_name].append(that_node)
        return 1, backrefs_map


# }}}


class TregexMatcher(TregexMatcherBase):
    @classmethod
    def match_node_descriptions(
        cls, descriptions: "NodeDescriptions", trees: List[Tree]
    ) -> Generator[Tree, Any, None]:
        is_negated = descriptions.is_negated
        use_basic_cat = descriptions.use_basic_cat

        for tree in trees:
            for node in tree.preorder_iter():
                for desc in descriptions:
                    if desc.condition_func(
                        node, desc.value, is_negated=is_negated, use_basic_cat=use_basic_cat
                    ):
                        yield node
                        break

    @classmethod
    def _match_and_conditions_cur_node(
        cls,
        this_node: Tree,
        this_name: Optional[str],
        and_conditions: AND_CONDITIONS,
    ) -> Tuple[int, dict]:
        match_count = 1
        backrefs_map: Dict[str, list] = {}

        # track names to prevent giving different nodes the same name in a
        # conjunction, imitating Stanford Tregex:
        # ```
        # echo "(A (B 1) (C 1))" | tregex.sh "A < B=n < C=n" -filter
        # ...
        # Error parsing expression: A < B=n < C=n
        # Parse exception: edu.stanford.nlp.trees.tregex.TregexParseException: Could not parse A < B=n < C=n
        # ```
        names = [this_name] if this_name is not None else []

        for and_condition in and_conditions:
            if isinstance(and_condition, NotAndCondition):
                match_count_cur_cond, backrefs_map_cur_cond = cls.match_not_and(
                    this_node, this_name, and_condition
                )
            else:
                that_name = and_condition.named_nodes.name
                if that_name is not None:
                    if and_condition.is_negated:
                        raise SystemExit(
                            "Error!!  It is not allowed to name a node that is under the scope"
                            f' of a negation operator. You need to remove the "{that_name}"'
                            " designation."
                        )
                    if that_name in names:
                        raise SystemExit(
                            f'Error!!  The name "{that_name}" has been assigned multiple times'
                            " in a conjunction."
                        )
                    else:
                        names.append(that_name)

                match_count_cur_cond, backrefs_map_cur_cond = cls.match_and(
                    this_node,
                    this_name,
                    and_condition,
                )

            if match_count_cur_cond == 0:
                return 0, {}

            match_count *= match_count_cur_cond
            for name, node_list in backrefs_map_cur_cond.items():
                backrefs_map[name] = node_list

        return match_count, backrefs_map

    @classmethod
    def match_and_conditions(
        cls,
        named_nodes: NamedNodes,
        and_conditions: AND_CONDITIONS,
    ) -> Tuple[List[Tree], dict]:
        res: List[Tree] = []
        backrefs_map: Dict[str, list] = {}

        this_name = named_nodes.name
        for this_node in named_nodes.nodes:
            match_count_cur_node, backrefs_map_cur_node = cls._match_and_conditions_cur_node(
                this_node, this_name, and_conditions
            )
            res += [this_node for _ in range(match_count_cur_node)]
            for name, nodes in backrefs_map_cur_node.items():
                backrefs_map[name] = backrefs_map.get(name, []) + nodes

        return res, backrefs_map

    @classmethod
    def match_or_conditions(
        cls,
        named_nodes: NamedNodes,
        or_conditions: List[AND_CONDITIONS],
    ) -> Tuple[List[Tree], dict]:
        res: List[Tree] = []
        backrefs_map: Dict[str, list] = {}
        for and_conditions in or_conditions:
            res_cur_and_conds, backrefs_map_cur_and_conds = cls.match_and_conditions(
                named_nodes, and_conditions
            )

            res.extend(res_cur_and_conds)
            for name, nodes in backrefs_map_cur_and_conds.items():
                backrefs_map[name] = backrefs_map.get(name, []) + nodes
        return res, backrefs_map


NodeDescription = namedtuple("NodeDescription", ("condition_func", "value"))


class NodeDescriptions:
    def __init__(
        self,
        node_descriptions: List[NodeDescription],
        *,
        is_negated: bool = False,
        use_basic_cat: bool = False,
    ):
        self.descriptions = node_descriptions
        self.is_negated = is_negated
        self.use_basic_cat = use_basic_cat

    def __iter__(self) -> Iterator[NodeDescription]:
        return iter(self.descriptions)

    def __repr__(self) -> str:
        string = "".join(desc.value for desc in self.descriptions)
        if self.use_basic_cat:
            string = "@" + string
        if self.is_negated:
            string = "!" + string
        return string

    def add_description(self, other_description: NodeDescription):
        self.descriptions.append(other_description)

    def toggle_negated(self):
        self.is_negated = not self.is_negated

    def toggle_use_basic_cat(self):
        self.use_basic_cat = not self.use_basic_cat


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
    t_REGEX = r"/[^/\n\r]*/[ix]*"
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

    def t_error(self, t):
        raise SystemExit(f'Tokenization error: Illegal character "{t.value[0]}"')

    literals = "{}"

    def __init__(self, tregex_pattern: str):
        self.lexer = lex.lex(module=self)
        self.lexer.input(tregex_pattern)

        self.backrefs_map: Dict[str, list] = {}
        self.pattern = tregex_pattern

    def findall(self, tree_string: str) -> List[Tree]:
        trees = Tree.fromstring(tree_string)
        parser = self.make_parser(trees)
        self._reset_lexer_state()

        return parser.parse(lexer=self.lexer)

    def get_nodes(self, name: str) -> List[Tree]:
        try:
            handled_nodes = self.backrefs_map[name]
        except KeyError:
            raise SystemExit(
                f'Error!!  There is no matched node "{name}"!  Did you specify such a'
                " label in the pattern?"
            )
        else:
            return handled_nodes

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
            ("right", "OR_NODE"),
            ("nonassoc", "EQUAL"),
        )

        # 1. node description
        def p_ID(p):
            """
            node_description : ID
            """
            # logging.debug("following rule: node_description -> ID")
            p[0] = NodeDescription(TregexMatcher.match_id, p[1])

        def p_REGEX(p):
            """
            node_description : REGEX
            """
            # logging.debug("following rule: node_description -> REGEX")
            p[0] = NodeDescription(TregexMatcher.match_regex, p[1])

        def p_BLANK(p):
            """
            node_description : BLANK
            """
            # logging.debug("following rule: node_description -> BLANK")
            p[0] = NodeDescription(TregexMatcher.match_blank, p[1])

        def p_not_node_descriptions(p):
            """
            node_descriptions : NOT node_descriptions
            """
            # logging.debug("following rule: node_descriptions -> NOT node_descriptions")
            p[2].toggle_negated()

            p[0] = p[2]

        def p_at_node_descriptions(p):
            """
            node_descriptions : AT node_descriptions
            """
            # logging.debug("following rule: node_descriptions -> AT node_descriptions")
            p[2].toggle_use_basic_cat()

            p[0] = p[2]

        def p_node_description(p):
            """
            node_descriptions : node_description
            """
            # logging.debug("following rule: node_descriptions -> node_description")
            p[0] = NodeDescriptions([p[1]])

        def p_node_descriptions_or_node_node_description(p):
            """
            node_descriptions : node_descriptions OR_NODE node_description
            """
            # logging.debug("following rule: node_descriptions -> node_descriptions OR_NODE node_description")
            p[1].add_description(p[3])

            p[0] = p[1]

        def p_node_descriptions(p):
            """
            named_nodes : node_descriptions
            """
            nodes = list(TregexMatcher.match_node_descriptions(p[1], trees))
            describing_str = str(p[1])
            logging.debug(f"following rule: named_nodes -> {describing_str}")

            p[0] = NamedNodes(None, nodes, describing_str)

        def p_lparen_node_description_rparen(p):
            """
            node_description : LPAREN node_description RPAREN
            """
            logging.debug("following rule: node_description -> LPAREN node_description RPAREN")
            p[0] = p[2]

        def p_lparen_node_descriptions_rparen(p):
            """
            node_descriptions : LPAREN node_descriptions RPAREN
            """
            logging.debug("following rule: node_descriptions -> LPAREN node_descriptions RPAREN")
            p[0] = p[2]

        def p_lparen_named_nodes_rparen(p):
            """
            named_nodes : LPAREN named_nodes RPAREN
            """
            logging.debug("following rule: named_nodes -> LPAREN named_nodes RPAREN")
            p[0] = p[2]

        def p_named_nodes_equal_id(p):
            """
            named_nodes : named_nodes EQUAL ID
            """
            name = p[3]
            named_nodes = p[1]
            logging.debug(f"following rule: {named_nodes.describing_str} = {name}")

            named_nodes.set_name(name)
            self.backrefs_map[name] = named_nodes.nodes

            p[0] = named_nodes

        # 2. relation
        # 2.1 RELATION
        def p_relation(p):
            """
            relation_data : RELATION
            """
            logging.debug("following rule: relation_data -> RELATION")
            rel_key = p[1]
            p[0] = RelationData(rel_key, self.RELATION_MAP[rel_key])

        # 2.2 REL_W_STR_ARG
        def p_rel_w_str_arg_lparen_named_nodes_rparen(p):
            """
            relation_data : REL_W_STR_ARG LPAREN named_nodes RPAREN
            """
            logging.debug(
                "following rule: relation_data -> REL_W_STR_ARG LPAREN named_nodes RPAREN"
            )
            rel_key = p[1]
            p[0] = RelationWithStrArgData(
                rel_key, self.REL_W_STR_ARG_MAP[rel_key], arg=p[3].nodes
            )

        # 2.3 REL_W_NUM_ARG
        def p_relation_number(p):
            """
            relation_data : RELATION NUMBER
            """
            logging.debug("following rule: relation_data -> RELATION NUMBER")
            rel_key, num = p[1:]
            if rel_key.endswith("-"):
                num = f"-{num}"
            p[0] = RelationWithNumArgData(rel_key, self.REL_W_NUM_ARG_MAP[rel_key], arg=int(num))

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

        def p_relation_data_equal_id(p):
            """
            and_conditions_backref : relation_data EQUAL ID
            """
            logging.debug("and_conditions_backref -> relation_data EUQAL ID")
            and_conditions_backref = []
            relation_data = p[1]
            name = p[3]
            those_nodes = self.get_nodes(name)

            for that_node in those_nodes:
                # backreferenced nodes should not be tracked by
                # self.backrefs_map, as they themselves are already the
                # tracking results, thus set "name" as None
                named_nodes = NamedNodes(name=None, nodes=[that_node])
                and_conditions_backref.append(
                    AndCondition(relation_data=relation_data, named_nodes=named_nodes)
                )

            p[0] = and_conditions_backref

        def p_and_conditions_and_conditions_backref(p):
            """
            and_conditions : and_conditions and_conditions_backref
            """
            logging.debug(
                "following rule: and_conditions -> and_conditions and_conditions_backref"
            )
            p[1].extend(p[2])
            p[0] = p[1]

        def p_and_conditions_backref(p):
            """
            and_conditions : and_conditions_backref
            """
            logging.debug("following rule: and_conditions -> and_conditions_backref")
            p[0] = p[1]

        def p_multi_relation_named_nodes(p):
            """
            and_conditions_multi_relation : MULTI_RELATION "{" named_nodes_list "}"
            """
            logging.debug(
                'following rule: and_conditions_multi_relation -> MULTI_RELATION "{"'
                ' named_nodes_list "}"'
            )
            rel_key = p[1]
            op = self.MULTI_RELATION_MAP[rel_key]
            named_nodes_list = p[3]

            conditions = []

            for i, named_nodes in enumerate(named_nodes_list, 1):
                multi_relation_data = MultiRelationData(rel_key, op, arg=i)
                conditions.append(
                    AndCondition(relation_data=multi_relation_data, named_nodes=named_nodes)
                )

            any_named_nodes = NamedNodes(
                None, list(node for tree in trees for node in tree.preorder_iter())
            )
            multi_relation_data = MultiRelationData(
                rel_key, op, arg=i + 1, is_negated=True  # type:ignore
            )
            conditions.append(
                AndCondition(relation_data=multi_relation_data, named_nodes=any_named_nodes)
            )

            p[0] = conditions

        def p_and_conditions_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions and_conditions_multi_relation
            """
            logging.debug(
                "following rule: and_conditions -> and_conditions and_conditions_multi_relation"
            )
            p[1].extend(p[2])
            p[0] = p[1]

        def p_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions_multi_relation
            """
            logging.debug("following rule: and_conditions -> and_conditions_multi_relation")
            p[0] = p[1]

        def p_not_and_conditions_multi_relation(p):
            """
            not_and_condition : NOT and_conditions_multi_relation
            """
            logging.debug(
                "following rule: not_and_condition -> NOT and_conditions_multi_relation"
            )
            and_conditions = p[2]

            not_and_condition = NotAndCondition(conditions=and_conditions)

            p[0] = not_and_condition

        def p_lparen_and_conditions_rparen(p):
            """
            and_conditions : LPAREN and_conditions RPAREN
            """
            logging.debug("following rule: and_conditions : LPAREN and_conditions RPAREN")
            p[0] = p[2]

        def p_and_conditions_or_and_conditions(p):
            """
            or_conditions : and_conditions OR_REL and_conditions
            """
            logging.debug(
                "following rule: or_conditions -> and_conditions OR_REL and_conditions"
            )

            p[0] = [p[1], p[3]]

        def p_and_conditions(p):
            """
            or_conditions : or_conditions OR_REL and_conditions
            """
            logging.debug("following rule: or_conditions -> or_conditions OR_REL and_conditions")
            p[1].append(p[2])

            p[0] = p[1]

        def p_lparen_or_conditions_rparen(p):
            """
            or_conditions : LPAREN or_conditions RPAREN
            """
            logging.debug("following rule: or_conditions -> LPAREN or_conditions RPAREN")
            p[0] = p[2]

        def p_not_lparen_and_conditions_rparen(p):
            """
            not_and_condition : NOT LPAREN and_conditions RPAREN
                              | NOT LBRACKET and_conditions RBRACKET
            """
            logging.debug(
                f"following rule: not_and_condition -> NOT {p[2]} and_conditions {p[4]}"
            )
            and_conditions = p[3]

            not_and_condition = NotAndCondition(conditions=and_conditions)

            p[0] = not_and_condition

        def p_not_lparen_or_conditions_rparen(p):
            """
            not_and_conditions : NOT LPAREN or_conditions RPAREN
                               | NOT LBRACKET or_conditions RBRACKET
            """
            logging.debug(
                f"following rule: not_and_conditions -> NOT {p[2]} or_conditions {p[4]}"
            )
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

        def p_named_nodes_and_conditions(p):
            """
            named_nodes : named_nodes and_conditions
            """
            logging.debug("following rule: named_nodes -> named_nodes and_conditions")
            named_nodes = p[1]
            or_conditions = p[2]

            res, backrefs_map = TregexMatcher.match_and_conditions(named_nodes, or_conditions)
            for name, node_list in backrefs_map.items():
                logging.debug(
                    "Mapping {} to nodes:\n  {}".format(
                        name, "\n  ".join(node.tostring() for node in node_list)
                    )
                )
                self.backrefs_map[name] = node_list

            named_nodes.set_nodes(res)
            p[0] = named_nodes

        def p_named_nodes_or_conditions(p):
            """
            named_nodes : named_nodes or_conditions
            """
            logging.debug("following rule: named_nodes -> named_nodes or_conditions")
            named_nodes = p[1]
            or_conditions = p[2]

            res, backrefs_map = TregexMatcher.match_or_conditions(named_nodes, or_conditions)
            for name, node_list in backrefs_map.items():
                logging.debug(
                    "Mapping {} to nodes:\n  {}".format(
                        name, "\n  ".join(node.tostring() for node in node_list)
                    )
                )
                self.backrefs_map[name] = node_list

            named_nodes.set_nodes(res)
            p[0] = named_nodes

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
                msg = (
                    f"{self.lexer.lexdata}\n{' ' * p.lexpos}˄\nParsing error at token"
                    f" '{p.value}'"
                )
            else:
                msg = "Parsing Error at EOF"
            raise SystemExit(msg)

        return yacc.yacc(debug=True, start="expr")
