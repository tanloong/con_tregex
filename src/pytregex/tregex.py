import logging
import re
from typing import Dict, List, Never

from condition import And, Condition, Not, Opt, Or
from node_descriptions import (
    NODE_ANY,
    NODE_ID,
    NODE_REGEX,
    NODE_ROOT,
    NamedNodes,
    NodeDescription,
    NodeDescriptions,
)
from ply import lex, yacc
from relation import *
from tree import Tree

logging.basicConfig(level=logging.DEBUG, filemode="w", format="%(filename)10s:%(lineno)4d:%(message)s")
log = logging.getLogger()


class TregexPattern:
    RELATION_MAP: dict[str, type[AbstractRelation]] = {
        "<": PARENT_OF,
        ">": CHILD_OF,
        "<<": DOMINATES,
        ">>": DOMINATED_BY,
        ">:": ONLY_CHILD_OF,
        "<:": HAS_ONLY_CHILD,
        ">`": LAST_CHILD_OF_PARENT,
        ">-": LAST_CHILD_OF_PARENT,
        "<`": PARENT_OF_LAST_CHILD,
        "<-": PARENT_OF_LAST_CHILD,
        ">,": LEFTMOST_CHILD_OF,
        "<,": HAS_LEFTMOST_CHILD,
        "<<`": HAS_RIGHTMOST_DESCENDANT,
        "<<-": HAS_RIGHTMOST_DESCENDANT,
        ">>`": RIGHTMOST_DESCENDANT_OF,
        ">>-": RIGHTMOST_DESCENDANT_OF,
        ">>,": LEFTMOST_DESCENDANT_OF,
        "<<,": HAS_LEFTMOST_DESCENDANT,
        "$..": LEFT_SISTER_OF,
        "$++": LEFT_SISTER_OF,
        "$--": RIGHT_SISTER_OF,
        "$,,": RIGHT_SISTER_OF,
        "$.": IMMEDIATE_LEFT_SISTER_OF,
        "$+": IMMEDIATE_LEFT_SISTER_OF,
        "$-": IMMEDIATE_RIGHT_SISTER_OF,
        "$,": IMMEDIATE_RIGHT_SISTER_OF,
        "$": SISTER_OF,
        "==": EQUALS,
        "<=": PARENT_EQUALS,
        "<<:": UNARY_PATH_ANCESTOR_OF,
        ">>:": UNARY_PATH_DESCEDANT_OF,
        ":": PATTERN_SPLITTER,
        ">#": IMMEDIATELY_HEADS,
        "<#": IMMEDIATELY_HEADED_BY,
        ">>#": HEADS,
        "<<#": HEADED_BY,
        "..": PRECEDES,
        ",,": FOLLOWS,
        ".": IMMEDIATELY_PRECEDES,
        ",": IMMEDIATELY_FOLLOWS,
        "<<<": ANCESTOR_OF_LEAF,
        "<<<-": ANCESTOR_OF_LEAF,
    }

    REL_W_STR_ARG_MAP: dict[str, type[AbstractRelation]] = {
        "<+": UNBROKEN_CATEGORY_DOMINATES,
        ">+": UNBROKEN_CATEGORY_IS_DOMINATED_BY,
        ".+": UNBROKEN_CATEGORY_PRECEDES,
        ",+": UNBROKEN_CATEGORY_FOLLOWS,
    }

    REL_W_NUM_ARG_MAP: dict[str, type[AbstractRelation]] = {
        ">": ITH_CHILD_OF,
        ">-": ITH_CHILD_OF,
        "<": HAS_ITH_CHILD,
        "<-": HAS_ITH_CHILD,
        "<<<": ANCESTOR_OF_ITH_LEAF,
        "<<<-": ANCESTOR_OF_ITH_LEAF,
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

    t_MULTI_RELATION = re.escape("<...")

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
        trees = tuple(Tree.fromstring(tree_string))
        parser = self.make_parser(trees)
        self._reset_lexer_state()

        # return parser.parse(lexer=self.lexer, debug=log)
        return parser.parse(lexer=self.lexer)

    def get_nodes(self, name: str) -> List[Tree]:
        try:
            handled_nodes = self.backrefs_map[name]
        except KeyError:
            raise SystemExit(
                f'Error!!  There is no matched node "{name}"!  Did you specify such a' " label in the pattern?"
            )
        else:
            return handled_nodes

    def _reset_lexer_state(self) -> None:
        """
        reset lexer.lexpos to make the lexer reusable
        https://github.com/dabeaz/ply/blob/master/doc/ply.md#internal-lexer-state
        """
        self.lexer.lexpos = 0

    def make_parser(self, trees: tuple[Tree, ...]):
        tokens = self.tokens

        precedence = (
            # keep consistency with Stanford Tregex
            # 1. "VP < NP < N" matches a VP which dominates both an NP and an N
            # 2. "VP < (NP < N)" matches a VP dominating an NP, which in turn dominates an N

            # shift on shift/reduce conflicts:
            # - node_descriptions_list -> node_descriptions_list node_descriptions .
            #  + condition -> . ! condition
            #  + condition -> . ! and_conditions_multi_relation
            #  + condition -> . ( condition )
            #  + condition -> . ( and_conditions )
            ("left", "NODEDESCS_AFTER_NODEDESCSLIST"),
            # shift on shift/reduce conflicts:
            # - node_descriptions -> node_descriptions or_conditions .
            # + or_conditions -> or_conditions . OR_REL and_conditions
            # - node_descriptions -> node_descriptions and_conditions .
            #  + and_conditions_multi_relation -> . MULTI_RELATION { node_descriptions_list }
            #  + condition -> . & condition
            #  + condition -> . ( and_conditions )
            #  + condition -> . ( condition )
            #  + condition -> . ( or_conditions )
            #  + condition -> . ? and_conditions_multi_relation
            #  + condition -> . ? condition
            #  + condition -> . [ and_conditions ]
            #  + condition -> . [ or_conditions ]
            #  + or_conditions -> and_conditions . OR_REL and_conditions
            #  + relation_data -> . RELATION
            #  + relation_data -> . RELATION NUMBER
            #  + relation_data -> . REL_W_STR_ARG ( node_descriptions )
            ("left", "ORCONDS_AFTER_NODEDESCS", "ANDCONDS_AFTER_NODEDESCS"),
            ("left", "?", "&", "(", "[", "OR_REL"),
            (
                "left",
                "RELATION",
                "REL_W_STR_ARG",
                "MULTI_RELATION",
            ),  # REL_W_NUM_ARG don't have to be declared, as they have already been as t_RELATION
            # shift on shift/reduce conflicts:
            # + condition -> ( condition . )
            # - and_conditions -> condition .
            # + node_description -> ( node_description . )
            # - node_descriptions -> node_description .
            # + node_descriptions -> node_descriptions . OR_NODE node_description
            #  - node_descriptions -> ! node_descriptions .
            #  - node_descriptions -> @ node_descriptions .
            # + condition -> . ! condition
            # - node_descriptions_list -> node_descriptions .
            ("left", "IMAGINE"),
            ("left", ")", "!", "@"),
            ("left", "OR_NODE"),
            # ("nonassoc", "="),
        )

        log_indent = 0

        # 1. node description
        def p_ID(p):
            """
            node_description : ID
            """
            p[0] = NodeDescription(NODE_ID, p[1])

        def p_REGEX(p):
            """
            node_description : REGEX
            """
            p[0] = NodeDescription(NODE_REGEX, p[1])

        def p_BLANK(p):
            """
            node_description : BLANK
            """
            p[0] = NodeDescription(NODE_ANY, p[1])

        def p_ROOT(p):
            """
            node_description : ROOT
            """
            p[0] = NodeDescription(NODE_ROOT, p[1])

        def p_not_node_descriptions(p):
            """
            node_descriptions : '!' node_descriptions
            """
            p[2].toggle_negated()
            p[2].set_string_repr(f"!{p[2].string_repr}")

            p[0] = p[2]

        def p_at_node_descriptions(p):
            """
            node_descriptions : '@' node_descriptions
            """
            p[2].set_use_basic_cat()
            p[2].set_string_repr(f"@{p[2].string_repr}")

            p[0] = p[2]

        def p_node_description(p):
            """
            node_descriptions : node_description %prec IMAGINE
            """
            p[0] = NodeDescriptions([p[1]])

        def p_node_descriptions_or_node_node_description(p):
            """
            node_descriptions : node_descriptions OR_NODE node_description
            """
            p[1].add_description(p[3])
            p[1].set_string_repr(f"{p[1].string_repr}{p[2]}{p[3].value}")

            p[0] = p[1]

        def p_lparen_node_description_rparen(p):
            """
            node_description : '(' node_description ')'
            """
            p[0] = p[2]

        def p_lparen_node_descriptions_rparen(p):
            """
            node_descriptions : '(' node_descriptions ')'
            """
            p[0] = p[2]

        # def p_named_nodes_equal_id(p):
        #     """
        #     named_nodes : named_nodes '=' ID
        #     """
        #     name = p[3]
        #     named_nodes = p[1]
        #
        #     named_nodes.set_name(name)
        #     self.backrefs_map[name] = named_nodes.nodes
        #
        #     p[0] = named_nodes

        # def p_link_id(p):
        #     """
        #     named_nodes : '~' ID
        #     """
        #     id = p[2]
        #     nodes = self.get_nodes(id)
        #
        #     p[0] = NamedNodes(None, nodes, string_repr=f"~ {id}")

        # 2. relation
        # 2.1 RELATION
        def p_relation(p):
            """
            relation_data : RELATION
            """
            string_repr = p[1]
            p[0] = RelationData(string_repr, self.RELATION_MAP[string_repr])

        # 2.2 REL_W_STR_ARG
        def p_rel_w_str_arg_lparen_node_descriptions_rparen(p):
            """
            relation_data : REL_W_STR_ARG '(' node_descriptions ')'
            """
            string_repr = p[1]
            p[0] = RelationWithStrArgData(string_repr, self.REL_W_STR_ARG_MAP[string_repr], arg=p[3])

        # 2.3 REL_W_NUM_ARG
        def p_relation_number(p):
            """
            relation_data : RELATION NUMBER
            """
            rel_key, num = p[1:]
            string_repr = f"{rel_key}{num}"

            if rel_key.endswith("-"):
                num = f"-{num}"
            p[0] = RelationWithNumArgData(string_repr, self.REL_W_NUM_ARG_MAP[rel_key], arg=int(num))

        def p_not_condition(p):
            """
            condition : '!' condition
            """
            p[0] = Not(p[2])

        def p_optional_condition(p):
            """
            condition : '?' condition
            """
            p[0] = Opt(p[2])

        # 3. and_conditions
        def p_relation_data_node_descriptions(p):
            """
            condition : relation_data node_descriptions %prec IMAGINE
            """
            # %prec IMAGINE: https://github.com/dabeaz/ply/issues/215
            # relation_data = p[1]
            # those_nodes, that_name = p[2].nodes, p[2].name

            # p[0] = ConditionOp(
            #     relation_data=relation_data, those_nodes=those_nodes, that_name=that_name
            # )
            p[0] = Condition(relation_data=p[1], node_descriptions=p[2])

        def p_and_condition(p):
            """
            condition : '&' condition
            """
            p[0] = p[2]

        def p_condition(p):
            """
            and_conditions : condition %prec IMAGINE
            """
            p[0] = And([p[1]])

        def p_and_conditions_condition(p):
            """
            and_conditions : and_conditions condition
            """
            p[1].append_condition(p[2])

            p[0] = p[1]

        def p_node_descriptions(p):
            """
            node_descriptions_list : node_descriptions %prec IMAGINE
                                   | node_descriptions ';'
            """
            p[0] = [p[1]]

        def p_node_descriptions_list_node_descriptions(p):
            """
            node_descriptions_list : node_descriptions_list node_descriptions %prec NODEDESCS_AFTER_NODEDESCSLIST
                                   | node_descriptions_list node_descriptions ';'
            """
            p[1].append(p[2])
            p[0] = p[1]

        def p_multi_relation_node_descriptions_list(p):
            """
            and_conditions_multi_relation : MULTI_RELATION "{" node_descriptions_list "}"
            """
            rel_key = p[1]
            rel_op = HAS_ITH_CHILD
            node_descriptions_list = p[3]

            conditions = []
            i = -1
            for i, node_descriptions in enumerate(node_descriptions_list, 1):
                multi_relation_data = RelationWithNumArgData(rel_key, rel_op, arg=i)
                conditions.append(
                    Condition(relation_data=multi_relation_data, node_descriptions=node_descriptions)
                )

            multi_relation_data = RelationWithNumArgData(rel_key, rel_op, arg=i + 1)
            node_descriptions = NodeDescriptions([NodeDescription(NODE_ANY, self.t_BLANK)])
            conditions.append(
                Not(Condition(relation_data=multi_relation_data, node_descriptions=node_descriptions))
            )

            p[0] = And(conditions)

        def p_and_conditions_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions and_conditions_multi_relation
            """
            p[0] = And([p[1], p[2]])

        def p_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions_multi_relation
            """
            p[0] = p[1]

        def p_not_and_conditions_multi_relation(p):
            """
            condition : '!' and_conditions_multi_relation
            """
            p[0] = Not(p[2])

        def p_optional_and_conditions_multi_relation(p):
            """
            condition : '?' and_conditions_multi_relation
            """
            p[0] = Opt(p[2])

        def p_lparen_and_condition_rparen(p):
            """
            condition : '(' condition ')'
            """
            p[0] = p[2]

        def p_and_conditions_or_and_conditions(p):
            """
            or_conditions : and_conditions OR_REL and_conditions
            """
            p[0] = Or([p[1], p[3]])

        def p_or_conditions_or_rel_and_conditions(p):
            """
            or_conditions : or_conditions OR_REL and_conditions
            """
            p[1].append_condition(p[3])

            p[0] = p[1]

        def p_lparen_or_conditions_rparen(p):
            """
            condition : '(' or_conditions ')'
                      | '[' or_conditions ']'
            """
            p[0] = p[2]

        def p_not_lparen_and_conditions_rparen(p):
            """
            condition : '(' and_conditions ')'
                      | '[' and_conditions ']'
            """
            p[0] = p[2]

        def p_node_descriptions_and_conditions(p):
            """
            node_descriptions : node_descriptions and_conditions %prec ANDCONDS_AFTER_NODEDESCS
            """
            p[1].set_condition(p[2])
            p[0] = p[1]

        # def p_node_descriptions_and_conditions(p):
        #     """
        #     nodes : node_descriptions and_conditions
        #     """
        #     node_descriptions, and_conditions = p[1:]
        #     nodes = [
        #         node
        #         for tree in trees
        #         for candidate in node_descriptions.searchNodeIterator(tree)
        #         for node in and_conditions.searchNodeIterator(candidate)
        #     ]
        #     p[0] = nodes

        def p_node_descriptions_or_conditions(p):
            """
            node_descriptions : node_descriptions or_conditions %prec ORCONDS_AFTER_NODEDESCS
            """
            p[1].set_condition(p[2])
            p[0] = p[1]

        # def p_node_descriptions_or_conditions(p):
        #     """
        #     nodes : node_descriptions or_conditions
        #     """
        #     node_descriptions, or_conditions = p[1:]
        #     nodes = [
        #         node
        #         for tree in trees
        #         for candidate in node_descriptions.searchNodeIterator(tree)
        #         for node in or_conditions.searchNodeIterator(candidate)
        #     ]
        #     p[0] = nodes

        def p_node_descriptions_list(p):
            """
            nodes : node_descriptions_list
            """
            nodes = []
            for tree in trees:
                for node_descriptions in p[1]:
                    nodes.extend(node_descriptions.searchNodeIterator(tree))
            p[0] = nodes

        def p_error(p) -> Never:
            if p:
                msg = f"{self.lexer.lexdata}\n{' ' * p.lexpos}˄\nParsing error at token '{p.value}'"
            else:
                msg = "Parsing Error at EOF"
            raise SystemExit(msg)

        return yacc.yacc(debug=True, start="nodes")
