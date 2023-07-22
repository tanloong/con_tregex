from abc import ABC, abstractmethod
from collections import namedtuple
import logging
import re
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Never,
    Optional,
)

from ply import lex, yacc

from condition import And, ConditionOp, Not, Opt, Or
from relation import Relation
from tree import Tree


class NamedNodes:
    def __init__(
        self, name: Optional[str], nodes: Optional[List[Tree]], string_repr: str = ""
    ) -> None:
        self.name = name
        self.nodes = nodes
        self.string_repr = string_repr

    def set_name(self, new_name: Optional[str]) -> None:
        self.name = new_name

    def set_nodes(self, new_nodes: List[Tree]) -> None:
        self.nodes = new_nodes


class AbstractRelationData(ABC):
    def __init__(self, string_repr: str, op: Callable):
        self.op = op
        self.string_repr = string_repr

    def __repr__(self) -> str:
        return self.string_repr

    def set_string_repr(self, s: str) -> None:
        self.string_repr = s

    @abstractmethod
    def condition_func(self, this_node: Tree, that_node: Tree):
        raise NotImplementedError()


class RelationData(AbstractRelationData):
    def __init__(self, string_repr: str, op: Callable) -> None:
        super().__init__(string_repr, op)

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node)


class RelationWithStrArgData(AbstractRelationData):
    def __init__(
        self,
        string_repr: str,
        op: Callable,
        *,
        arg: List[Tree],
    ) -> None:
        super().__init__(string_repr, op)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class RelationWithNumArgData(AbstractRelationData):
    def __init__(
        self,
        string_repr: str,
        op: Callable,
        *,
        arg: int,
    ) -> None:
        super().__init__(string_repr, op)
        self.arg = arg

    def condition_func(self, this_node: Tree, that_node: Tree) -> bool:
        return self.op(this_node, that_node, self.arg)


class MultiRelationData(RelationWithNumArgData):
    def __init__(
        self,
        string_repr: str,
        op: Callable,
        *,
        arg: int,
    ) -> None:
        super().__init__(string_repr, op, arg=arg)


class TregexMatcher:
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
        value: str = "",
        *,
        is_negated: bool = False,
        use_basic_cat: bool = False,
    ) -> bool:
        return not is_negated

    @classmethod
    def match_root(
        cls,
        node: Tree,
        value: str = "",
        *,
        is_negated: bool = False,
        use_basic_cat: bool = False,
    ) -> bool:
        if node.parent is None:
            return True
        return False

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


NodeDescription = namedtuple("NodeDescription", ("condition_func", "value"))


class NodeDescriptions:
    def __init__(
        self,
        node_descriptions: List[NodeDescription],
        *,
        is_negated: bool = False,
        use_basic_cat: bool = False,
    ) -> None:
        self.descriptions = node_descriptions
        self.is_negated = is_negated
        self.use_basic_cat = use_basic_cat

        self.string_repr = "".join(desc.value for desc in self.descriptions)

    def __iter__(self) -> Iterator[NodeDescription]:
        return iter(self.descriptions)

    def __repr__(self) -> str:
        return self.string_repr

    def set_string_repr(self, s: str):
        self.string_repr = s

    def add_description(self, other_description: NodeDescription) -> None:
        self.descriptions.append(other_description)

    def toggle_negated(self) -> None:
        self.is_negated = not self.is_negated

    def toggle_use_basic_cat(self) -> None:
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
        "OR_NODE",
        "OR_REL",
        "NUMBER",
        "ID",
        "ROOT",
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
    t_OR_REL = r"\|\|"
    t_OR_NODE = r"\|"
    t_NUMBER = r"[0-9]+"
    t_ID = r"[^ 0-9\n\r(/|@!#&)=?[\]><~_.,$:{};][^ \n\r(/|@!#&)=?[\]><~.$:{};]*"
    t_ROOT = r"_ROOT_"
    t_ignore = " \r\t"

    def t_error(self, t) -> Never:
        raise SystemExit(f'Tokenization error: Illegal character "{t.value[0]}"')

    literals = "!?()[]{}@&=~;"

    def __init__(self, tregex_pattern: str) -> None:
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

    def _reset_lexer_state(self) -> None:
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
            # https://github.com/dabeaz/ply/issues/215
            ("left", "IMAGINE_REDUCE"),
            ("left", "OR_REL"),
            ("right", "OR_NODE"),
            ("nonassoc", "="),
        )

        log_indent = 0

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

        def p_ROOT(p):
            """
            node_description : ROOT
            """
            # logging.debug("following rule: node_description -> ROOT")
            p[0] = NodeDescription(TregexMatcher.match_root, p[1])

        def p_not_node_descriptions(p):
            """
            node_descriptions : '!' node_descriptions
            """
            # logging.debug("following rule: node_descriptions -> ! node_descriptions")
            p[2].toggle_negated()
            p[2].set_string_repr(f"!{p[2].string_repr}")

            p[0] = p[2]

        def p_at_node_descriptions(p):
            """
            node_descriptions : '@' node_descriptions
            """
            # logging.debug("following rule: node_descriptions -> @ node_descriptions")
            p[2].toggle_use_basic_cat()
            p[2].set_string_repr(f"@{p[2].string_repr}")

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
            string_repr = p[1].string_repr
            logging.debug(f"following rule: named_nodes -> {string_repr}")

            p[0] = NamedNodes(None, nodes, string_repr)

        def p_lparen_node_description_rparen(p):
            """
            node_description : '(' node_description ')'
            """
            # logging.debug("following rule: node_description -> ( node_description )")
            p[0] = p[2]

        def p_lparen_node_descriptions_rparen(p):
            """
            node_descriptions : '(' node_descriptions ')'
            """
            # logging.debug("following rule: node_descriptions -> ( node_descriptions )")
            p[0] = p[2]

        def p_lparen_named_nodes_rparen(p):
            """
            named_nodes : '(' named_nodes ')'
            """
            # logging.debug("following rule: named_nodes -> ( named_nodes )")
            p[0] = p[2]

        def p_named_nodes_equal_id(p):
            """
            named_nodes : named_nodes '=' ID
            """
            name = p[3]
            named_nodes = p[1]
            logging.debug(f"following rule: {named_nodes.string_repr} = {name}")

            named_nodes.set_name(name)
            self.backrefs_map[name] = named_nodes.nodes

            p[0] = named_nodes

        def p_link_id(p):
            """
            named_nodes : '~' ID
            """
            logging.debug("following rule: named_nodes -> '~' ID")
            id = p[2]
            nodes = self.get_nodes(id)

            p[0] = NamedNodes(None, nodes, string_repr=f"~ {id}")

        # 2. relation
        # 2.1 RELATION
        def p_relation(p):
            """
            relation_data : RELATION
            """
            # logging.debug("following rule: relation_data -> RELATION")
            string_repr = p[1]
            p[0] = RelationData(string_repr, self.RELATION_MAP[string_repr])

        # 2.2 REL_W_STR_ARG
        def p_rel_w_str_arg_lparen_named_nodes_rparen(p):
            """
            relation_data : REL_W_STR_ARG '(' named_nodes ')'
            """
            # logging.debug("following rule: relation_data -> REL_W_STR_ARG ( named_nodes )")
            string_repr = p[1]
            p[0] = RelationWithStrArgData(
                string_repr, self.REL_W_STR_ARG_MAP[string_repr], arg=p[3].nodes
            )

        # 2.3 REL_W_NUM_ARG
        def p_relation_number(p):
            """
            relation_data : RELATION NUMBER
            """
            # logging.debug("following rule: relation_data -> RELATION NUMBER")
            rel_key, num = p[1:]
            string_repr = f"{rel_key}{num}"

            if rel_key.endswith("-"):
                num = f"-{num}"
            p[0] = RelationWithNumArgData(
                string_repr, self.REL_W_NUM_ARG_MAP[rel_key], arg=int(num)
            )

        def p_not_and_condition(p):
            """
            and_condition : '!' and_condition
            """
            logging.debug("following rule: and_condition -> ! and_condition")
            p[0] = Not(p[2])

        def p_optional_and_condition(p):
            """
            and_condition : '?' and_condition
            """
            logging.debug("following rule: and_condition -> ? and_condition")
            p[0] = Opt(p[2])

        # 3. and_conditions
        def p_relation_data_named_nodes(p):
            """
            and_condition : relation_data named_nodes %prec IMAGINE_REDUCE
            """
            # %prec IMAGINE_REDUCE: https://github.com/dabeaz/ply/issues/215
            logging.debug(
                f"following rule: and_condition -> {p[1].string_repr} {p[2].string_repr}"
            )
            relation_data = p[1]
            those_nodes, that_name = p[2].nodes, p[2].name

            p[0] = ConditionOp(
                relation_data=relation_data, those_nodes=those_nodes, that_name=that_name
            )

        def p_and_and_condition(p):
            """
            and_condition : '&' and_condition
            """
            # logging.debug("following rule: and_condition -> & and_condition")
            p[0] = p[2]

        def p_and_conditions_and_condition(p):
            """
            and_conditions : and_conditions and_condition
            """
            # logging.debug("following rule: and_conditions -> and_conditions and_condition")
            p[1].append_condition(p[2])

            p[0] = p[1]

        def p_and_condition(p):
            """
            and_conditions : and_condition
            """
            p[0] = And([p[1]])

        def p_multi_relation_named_nodes(p):
            """
            and_conditions_multi_relation : MULTI_RELATION "{" named_nodes_list "}"
            """
            rel_key = p[1]
            op = self.MULTI_RELATION_MAP[rel_key]
            named_nodes_list = p[3]
            logging.debug(
                f"following rule: and_conditions_multi_relation -> {rel_key} {{"
                " named_nodes_list }"
            )

            conditions = []

            for i, named_nodes in enumerate(named_nodes_list, 1):
                multi_relation_data = MultiRelationData(rel_key, op, arg=i)
                those_nodes, that_name = named_nodes.nodes, named_nodes.name
                conditions.append(
                    ConditionOp(
                        relation_data=multi_relation_data,
                        those_nodes=those_nodes,
                        that_name=that_name,
                    )
                )

            any_nodes = list(node for tree in trees for node in tree.preorder_iter())
            multi_relation_data = MultiRelationData(rel_key, op, arg=i + 1)  # type:ignore
            conditions.append(
                Not(
                    ConditionOp(
                        relation_data=multi_relation_data, those_nodes=any_nodes, that_name=None
                    )
                )
            )

            p[0] = And(conditions)

        def p_and_conditions_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions and_conditions_multi_relation
            """
            logging.debug(
                "following rule: and_conditions -> and_conditions and_conditions_multi_relation"
            )
            p[0] = And([p[1], p[2]])

        def p_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions_multi_relation
            """
            logging.debug("following rule: and_conditions -> and_conditions_multi_relation")
            p[0] = p[1]

        def p_not_and_conditions_multi_relation(p):
            """
            and_condition : '!' and_conditions_multi_relation
            """
            logging.debug("following rule: and_condition -> ! and_conditions_multi_relation")
            p[0] = Not(p[2])

        def p_optional_and_conditions_multi_relation(p):
            """
            and_condition : '?' and_conditions_multi_relation
            """
            logging.debug(
                "following rule: optional_and_condition -> ? and_conditions_multi_relation"
            )
            p[0] = Opt(p[2])

        def p_lparen_and_condition_rparen(p):
            """
            and_condition : '(' and_condition ')'
            """
            logging.debug("following rule: and_condition : ( and_condition )")
            p[0] = p[2]

        def p_lparen_and_conditions_rparen(p):
            """
            and_conditions : '(' and_conditions ')'
            """
            logging.debug("following rule: and_conditions : ( and_conditions )")
            p[0] = p[2]

        def p_and_conditions_or_and_conditions(p):
            """
            or_conditions : and_conditions OR_REL and_conditions
            """
            logging.debug(
                f"following rule: or_conditions -> and_conditions {p[2]} and_conditions"
            )

            p[0] = Or([p[1], p[3]])

        def p_or_conditions_or_rel_and_conditions(p):
            """
            or_conditions : or_conditions OR_REL and_conditions
            """
            logging.debug(
                f"following rule: or_conditions -> or_conditions {p[2]} and_conditions"
            )
            p[1].append_condition(p[2])

            p[0] = p[1]

        def p_lparen_or_conditions_rparen(p):
            """
            or_conditions : '(' or_conditions ')'
            """
            logging.debug("following rule: or_conditions -> ( or_conditions )")
            p[0] = p[2]

        def p_not_lparen_and_conditions_rparen(p):
            """
            and_condition : '!' '(' and_conditions ')'
                          | '!' '[' and_conditions ']'
            """
            logging.debug(f"following rule: and_condition -> ! {p[2]} and_conditions {p[4]}")
            p[0] = Not(p[3])

        def p_optional_lparen_and_conditions_rparen(p):
            """
            and_condition : '?' '(' and_conditions ')'
                          | '?' '[' and_conditions ']'
            """
            logging.debug(f"following rule: and_condition -> ? {p[2]} and_conditions {p[4]}")
            p[0] = Opt(p[3])

        def p_not_lparen_or_conditions_rparen(p):
            """
            and_condition : '!' '(' or_conditions ')'
                          | '!' '[' or_conditions ']'
            """
            logging.debug(f"following rule: not_and_conditions -> ! {p[2]} or_conditions {p[4]}")
            p[0] = Not(p[3])

        def p_optional_lparen_or_conditions_rparen(p):
            """
            and_condition : '?' '(' or_conditions ')'
                          | '?' '[' or_conditions ']'
            """
            logging.debug(f"following rule: and_condition -> ? {p[2]} or_conditions {p[4]}")
            p[0] = Opt(p[3])

        def p_lbracket_or_conditions_rbracket(p):
            """
            or_conditions : '[' or_conditions ']'
            """
            logging.debug("following rule: or_conditions -> [ or_conditions ]")
            p[0] = p[2]

        def p_named_nodes_and_conditions(p):
            """
            named_nodes : named_nodes and_conditions
            """
            logging.debug("following rule: named_nodes -> named_nodes and_conditions")
            named_nodes = p[1]
            and_conditions = p[2]

            matched_nodes, backrefs_map = and_conditions.match(
                these_nodes=named_nodes.nodes, this_name=named_nodes.name
            )
            self.backrefs_map.update(backrefs_map)

            named_nodes.set_nodes(matched_nodes)
            p[0] = named_nodes

        def p_named_nodes_or_conditions(p):
            """
            named_nodes : named_nodes or_conditions
            """
            logging.debug("following rule: named_nodes -> named_nodes or_conditions")
            named_nodes = p[1]
            or_conditions = p[2]

            matched_nodes, backrefs_map = or_conditions.match(
                these_nodes=named_nodes.nodes, this_name=named_nodes.name
            )
            self.backrefs_map.update(backrefs_map)

            named_nodes.set_nodes(matched_nodes)
            p[0] = named_nodes

        def p_named_nodes(p):
            """
            named_nodes_list : named_nodes
                             | named_nodes ';'
            """
            logging.debug("following rule: named_nodes_list -> named_nodes")
            # List[List[Tree]]
            p[0] = [p[1]]

        def p_named_nodes_list_named_nodes(p):
            """
            named_nodes_list : named_nodes_list named_nodes
                             | named_nodes_list named_nodes ';'
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

        def p_error(p) -> Never:
            if p:
                msg = (
                    f"{self.lexer.lexdata}\n{' ' * p.lexpos}˄\nParsing error at token"
                    f" '{p.value}'"
                )
            else:
                msg = "Parsing Error at EOF"
            raise SystemExit(msg)

        return yacc.yacc(debug=True, start="expr")
