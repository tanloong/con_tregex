import logging
import re
import warnings
from typing import List, Never

from . import relation as _r
from .condition import (
    NODE_ANY,
    NODE_ID,
    NODE_REGEX,
    NODE_ROOT,
    AbstractCondition,
    And,
    BackRef,
    Condition,
    NodeDescription,
    NodeDescriptions,
    Not,
    Opt,
    Or,
)
from .exceptions import ParseException
from .ply import lex, yacc
from .tree import Tree


class TregexPattern:
    RELATION_MAP: dict[str, type[_r.AbstractRelation]] = {
        "<": _r.PARENT_OF,
        ">": _r.CHILD_OF,
        "<<": _r.DOMINATES,
        ">>": _r.DOMINATED_BY,
        ">:": _r.ONLY_CHILD_OF,
        "<:": _r.HAS_ONLY_CHILD,
        ">`": _r.LAST_CHILD_OF_PARENT,
        ">-": _r.LAST_CHILD_OF_PARENT,
        "<`": _r.PARENT_OF_LAST_CHILD,
        "<-": _r.PARENT_OF_LAST_CHILD,
        ">,": _r.LEFTMOST_CHILD_OF,
        "<,": _r.HAS_LEFTMOST_CHILD,
        "<<`": _r.HAS_RIGHTMOST_DESCENDANT,
        "<<-": _r.HAS_RIGHTMOST_DESCENDANT,
        ">>`": _r.RIGHTMOST_DESCENDANT_OF,
        ">>-": _r.RIGHTMOST_DESCENDANT_OF,
        ">>,": _r.LEFTMOST_DESCENDANT_OF,
        "<<,": _r.HAS_LEFTMOST_DESCENDANT,
        "$..": _r.LEFT_SISTER_OF,
        "$++": _r.LEFT_SISTER_OF,
        "$--": _r.RIGHT_SISTER_OF,
        "$,,": _r.RIGHT_SISTER_OF,
        "$.": _r.IMMEDIATE_LEFT_SISTER_OF,
        "$+": _r.IMMEDIATE_LEFT_SISTER_OF,
        "$-": _r.IMMEDIATE_RIGHT_SISTER_OF,
        "$,": _r.IMMEDIATE_RIGHT_SISTER_OF,
        "$": _r.SISTER_OF,
        "==": _r.EQUALS,
        "<=": _r.PARENT_EQUALS,
        "<<:": _r.UNARY_PATH_ANCESTOR_OF,
        ">>:": _r.UNARY_PATH_DESCEDANT_OF,
        ":": _r.PATTERN_SPLITTER,
        ">#": _r.IMMEDIATELY_HEADS,
        "<#": _r.IMMEDIATELY_HEADED_BY,
        ">>#": _r.HEADS,
        "<<#": _r.HEADED_BY,
        "..": _r.PRECEDES,
        ",,": _r.FOLLOWS,
        ".": _r.IMMEDIATELY_PRECEDES,
        ",": _r.IMMEDIATELY_FOLLOWS,
        "<<<": _r.ANCESTOR_OF_LEAF,
        "<<<-": _r.ANCESTOR_OF_LEAF,
    }

    REL_W_STR_ARG_MAP: dict[str, type[_r.AbstractRelation]] = {
        "<+": _r.UNBROKEN_CATEGORY_DOMINATES,
        ">+": _r.UNBROKEN_CATEGORY_IS_DOMINATED_BY,
        ".+": _r.UNBROKEN_CATEGORY_PRECEDES,
        ",+": _r.UNBROKEN_CATEGORY_FOLLOWS,
    }

    REL_W_NUM_ARG_MAP: dict[str, type[_r.AbstractRelation]] = {
        ">": _r.ITH_CHILD_OF,
        ">-": _r.ITH_CHILD_OF,
        "<": _r.HAS_ITH_CHILD,
        "<-": _r.HAS_ITH_CHILD,
        "<<<": _r.ANCESTOR_OF_ITH_LEAF,
        "<<<-": _r.ANCESTOR_OF_ITH_LEAF,
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

        self.pattern = tregex_pattern
        # > keep track of which variables we've seen, so that we can reject
        # > some nonsense patterns such as ones that reset variables or link
        # > to variables that haven't been set
        self.backref_table: dict[str, BackRef] = {}

    def findall(self, tree_string: str) -> List[Tree]:
        # TODO: must tupleize?
        trees = tuple(Tree.fromstring(tree_string))
        parser = self.make_parser(trees)
        self._reset_lexer_state()

        return parser.parse(lexer=self.lexer, debug=(logging.getLogger().level == logging.DEBUG))
        # return parser.parse(lexer=self.lexer)

    def get_nodes(self, name: str) -> List[Tree]:
        try:
            backref = self.backref_table[name]
        except KeyError as e:
            raise SystemExit(
                f'Error!!  There is no matched node "{name}"!  Did you specify such a label in the pattern?'
            ) from e
        else:
            assert backref.nodes is not None
            return backref.nodes

    def _reset_lexer_state(self) -> None:
        """
        reset lexer.lexpos to make the lexer reusable
        https://github.com/dabeaz/ply/blob/master/doc/ply.md#internal-lexer-state
        """
        self.lexer.lexpos = 0

    def make_parser(self, trees: tuple[Tree, ...]):
        tokens = self.tokens

        precedence = (
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
            ("nonassoc", "=", "~"),
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
            if not p[2].negate():
                # Use warnings.warn instead of logging.warning to suppress repetitions
                warnings.warn("repeated '!'", category=SyntaxWarning, stacklevel=1)

            p[0] = p[2]

        def p_at_node_descriptions(p):
            """
            node_descriptions : '@' node_descriptions
            """
            if not p[2].enable_basic_cat():
                # Use warnings.warn instead of logging.warning to suppress repetitions
                warnings.warn("repeated '@'", category=SyntaxWarning, stacklevel=1)

            p[0] = p[2]

        def p_node_description(p):
            """
            node_descriptions : node_description %prec IMAGINE
            """
            p[0] = NodeDescriptions(p[1])

        def p_node_descriptions_or_node_node_description(p):
            """
            node_descriptions : node_descriptions OR_NODE node_description
            """
            p[1].add_description(p[3])

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

        def p_node_descriptions_equal_id(p):
            """
            node_descriptions : node_descriptions '=' ID
            """
            name: str = p[3]
            node_descriptions: NodeDescriptions = p[1]
            # '!' has higher precedence than '=', it will have already been
            # reduced to node_descriptions prior to '='
            if node_descriptions.under_negation:
                raise ParseException("No named tregex nodes allowed in the scope of negation")

            backref = BackRef(node_descriptions, None)
            self.backref_table[name] = backref
            node_descriptions.set_backref(backref, name)

            p[0] = node_descriptions

        def p_link_id(p):
            """
            node_descriptions : '~' ID
            """
            linked_name: str = p[2]
            if linked_name not in self.backref_table:
                raise ParseException(f"Variable {linked_name} was referenced before it was declared")

            orig_nodedescs = self.backref_table[linked_name].node_descriptions
            node_descriptions = NodeDescriptions(
                *orig_nodedescs.descriptions,
                under_negation=orig_nodedescs.under_negation,
                use_basic_cat=orig_nodedescs.use_basic_cat,
            )

            p[0] = node_descriptions

        # def p_equal_id(p):
        #     """
        #     node_descriptions : '=' ID
        #     """
        #     name = p[2]
        #     if name not in self.backref_table:
        #         raise ParseException(f"Variable {name} was referenced before it was declared")
        #
        #     backref = self.backref_table[name]
        #     node_descriptions = NodeDescriptions(backref=backref, name=name)
        #     p[0] = node_descriptions

        # 2. relation
        # 2.1 RELATION
        def p_relation(p):
            """
            relation_data : RELATION
            """
            symbol = p[1]
            p[0] = _r.RelationData(self.RELATION_MAP[symbol], symbol)

        # 2.2 REL_W_STR_ARG
        def p_rel_w_str_arg_lparen_node_descriptions_rparen(p):
            """
            relation_data : REL_W_STR_ARG '(' node_descriptions ')'
            """
            symbol = p[1]
            p[0] = _r.RelationWithStrArgData(self.REL_W_STR_ARG_MAP[symbol], symbol, arg=p[3])

        # 2.3 REL_W_NUM_ARG
        def p_relation_number(p):
            """
            relation_data : RELATION NUMBER
            """
            rel_key, num = p[1:]
            symbol = f"{rel_key}{num}"

            if rel_key.endswith("-"):
                num = f"-{num}"
            p[0] = _r.RelationWithNumArgData(self.REL_W_NUM_ARG_MAP[rel_key], symbol, arg=int(num))

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
            p[0] = And(p[1])

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
            rel_key = "<"
            rel_op = _r.HAS_ITH_CHILD
            node_descriptions_list = p[3]

            conditions: list[AbstractCondition] = []
            for i, node_descriptions in enumerate(node_descriptions_list, 1):
                multi_relation_data = _r.RelationWithNumArgData(rel_op, rel_key, arg=i)
                conditions.append(
                    Condition(relation_data=multi_relation_data, node_descriptions=node_descriptions)
                )

            multi_relation_data = _r.RelationWithNumArgData(rel_op, rel_key, arg=i + 1)
            node_descriptions = NodeDescriptions(NodeDescription(NODE_ANY, self.t_BLANK))
            conditions.append(
                Not(Condition(relation_data=multi_relation_data, node_descriptions=node_descriptions))
            )

            p[0] = And(*conditions)

        def p_and_conditions_and_conditions_multi_relation(p):
            """
            and_conditions : and_conditions and_conditions_multi_relation
            """
            p[0] = And(p[1], p[2])

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
            p[0] = Or(p[1], p[3])

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
            if p is None:
                msg = "Parsing Error at EOF"
            else:
                msg = f"{self.lexer.lexdata}\n{' ' * p.lexpos}˄\nParsing error at token '{p.value}'"
            raise SystemExit(msg)

        return yacc.yacc(debug=False, start="nodes")
