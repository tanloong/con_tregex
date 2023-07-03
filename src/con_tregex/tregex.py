# /home/tan/.local/share/stanford-tregex-2020-11-17/stanford-tregex-4.2.0-sources/edu/stanford/nlp/trees/tregex/Relation.java
# https://github.com/stanfordnlp/CoreNLP/blob/efc66a9cf49fecba219dfaa4025315ad966285cc/test/src/edu/stanford/nlp/trees/tregex/TregexTest.java
import re
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from .ply import lex, yacc
from .relation import Relation
from .tree import Tree

NAMED_NODES = Tuple[List[Tree], Optional[str]]


class TregexMatcherBase:  # {{{
    @classmethod
    def match_condition(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
        condition_func: Callable,
    ) -> Tuple[NAMED_NODES, dict]:
        if modifier is None:
            res, backrefs_map = cls._exactly_match_condition(these, those, condition_func)
        elif modifier == "!":
            res = cls._not_match_condition(these, those, condition_func)
            backrefs_map = {}
        else:
            res = these[0]
            backrefs_map = cls._optionally_match_condition(these, those, condition_func)
        this_name = these[1]
        return ((res, this_name), backrefs_map)

    @classmethod
    def _exactly_match_condition(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> Tuple[List[Tree], dict]:
        these_nodes, this_name = these
        those_nodes, that_name = those

        backrefs_map: Dict[str, list] = {}
        if this_name is not None:
            backrefs_map[this_name] = []
        if that_name is not None:
            backrefs_map[that_name] = []

        res = []
        for this_node in these_nodes:
            for that_node in those_nodes:
                if condition_func(this_node, that_node):
                    if this_name is not None:
                        backrefs_map[this_name].append(this_node)
                    if that_name is not None:
                        backrefs_map[that_name].append(that_node)
                    res.append(this_node)

        return (res, backrefs_map)

    @classmethod
    def _not_match_condition(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        condition_func: Callable,
    ) -> List[Tree]:
        these_nodes, _ = these
        those_nodes, _ = those

        res = []
        for this_node in these_nodes:
            if all(
                map(
                    lambda that_node, this_node=this_node: not condition_func(  # type:ignore
                        this_node, that_node
                    ),
                    those_nodes,
                )
            ):
                res.append(this_node)
        return res

    @classmethod
    def _optionally_match_condition(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        condition_func: Callable[[Tree, Tree], bool],
    ) -> dict:
        these_nodes, this_name = these
        those_nodes, that_name = those

        if that_name is None:
            return {}

        backrefs_map: Dict[str, list] = {}
        backrefs_map[that_name] = []
        for this_node in these_nodes:
            for that_node in those_nodes:
                if condition_func(this_node, that_node):
                    backrefs_map[that_name].append(that_node)
        return backrefs_map


# }}}
class TregexMatcher(TregexMatcherBase):  # {{{
    @classmethod
    def match_any(cls, trees: List[Tree]) -> Generator[Tree, Any, None]:
        for tree in trees:
            for node in tree.preorder_iter():
                yield node

    @classmethod
    def match_label(
        cls, trees: List[Tree], label: str, is_negate: bool = False
    ) -> Generator[Tree, Any, None]:
        if not is_negate:
            for tree in trees:
                for node in tree.preorder_iter():
                    if node.label == label:
                        yield node
        else:
            for tree in trees:
                for node in tree.preorder_iter():
                    if node.label != label:
                        yield node

    @classmethod
    def match_regex(
        cls, trees: List[Tree], regex: str, is_negate: bool = False
    ) -> Generator[Tree, Any, None]:
        pattern = re.compile(regex)
        if not is_negate:
            for tree in trees:
                for node in tree.preorder_iter():
                    if node.label is None:
                        continue
                    if pattern.search(node.label) is not None:
                        yield node
        else:
            for tree in trees:
                for node in tree.preorder_iter():
                    if node.label is None:
                        yield node
                        continue
                    if pattern.search(node.label) is None:
                        yield node

    @classmethod
    def parent_of(
        cls,
        these: Tuple[List[Tree], str],
        those: Tuple[List[Tree], str],
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.parent_of(this_node, that_node),
        )

    @classmethod
    def child_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.child_of(this_node, that_node),
        )

    @classmethod
    def dominates(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.dominates(this_node, that_node),
        )

    # }}}
    @classmethod
    def dominated_by(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.dominated_by(this_node, that_node),
        )

    @classmethod
    def only_child_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.only_child_of(this_node, that_node),
        )

    @classmethod
    def has_only_child(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.has_only_child(this_node, that_node),
        )

    @classmethod
    def last_child_of_parent(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.last_child_of_parent(this_node, that_node),
        )

    @classmethod
    def parent_of_last_child(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.parent_of_last_child(this_node, that_node),
        )

    @classmethod
    def leftmost_child_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.leftmost_child_of(this_node, that_node),
        )

    @classmethod
    def has_leftmost_child(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.has_leftmost_child(this_node, that_node),
        )

    @classmethod
    def has_rightmost_descendant(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.has_rightmost_descendant(this_node, that_node),
        )

    @classmethod
    def rightmost_descendant_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.rightmost_descendant_of(this_node, that_node),
        )

    @classmethod
    def leftmost_descendant_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.leftmost_descendant_of(this_node, that_node),
        )

    @classmethod
    def has_leftmost_descendant(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.has_leftmost_descendant(this_node, that_node),
        )

    @classmethod
    def left_sister_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.left_sister_of(this_node, that_node),
        )

    @classmethod
    def right_sister_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.right_sister_of(this_node, that_node),
        )

    @classmethod
    def immediate_left_sister_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.immediate_left_sister_of(this_node, that_node),
        )

    @classmethod
    def immediate_right_sister_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.immediate_right_sister_of(
                this_node, that_node
            ),
        )

    @classmethod
    def sister_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.sister_of(this_node, that_node),
        )

    @classmethod
    def equals(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.equals(this_node, that_node),
        )

    @classmethod
    def parent_equals(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.parent_equals(this_node, that_node),
        )

    @classmethod
    def unary_path_ancestor_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.unary_path_ancestor_of(this_node, that_node),
        )

    @classmethod
    def unary_path_descedant_of(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.unary_path_descedant_of(this_node, that_node),
        )

    @classmethod
    def pattern_splitter(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(these, those, modifier, lambda this_node, that_node: True)

    @classmethod
    def immediately_heads(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.immediately_heads(this_node, that_node),
        )

    @classmethod
    def immediately_headed_by(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.immediately_headed_by(this_node, that_node),
        )

    @classmethod
    def heads(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.heads(this_node, that_node),
        )

    @classmethod
    def headed_by(
        cls,
        these: NAMED_NODES,
        those: NAMED_NODES,
        modifier: Optional[str],
    ) -> Tuple[NAMED_NODES, dict]:
        return cls.match_condition(
            these,
            those,
            modifier,
            lambda this_node, that_node: Relation.headed_by(this_node, that_node),
        )

    @classmethod
    def and_(
        cls,
        these: NAMED_NODES,
        and_conditions: Tuple[Tuple[Callable, NAMED_NODES, Optional[str]]],
    ) -> Tuple[NAMED_NODES, dict]:
        """
        :and_conditions: a tuple of (func, those_nodes, is_negate), func is a TregexMatcher method, e.g, TregexMatcher.parent_of
        """
        backrefs_map = {}
        for func, those, modifier in and_conditions:
            assert modifier in (None, "!", "?")
            these, backrefs_map = func(these, those, modifier)
            if not these[0]:
                break
        return (these, backrefs_map)


class TregexPattern:
    tokens = [  # {{{
        "ID",
        "REGEX",
        "BLANK",
        "RELATION",
        "NEGATION",
        "OPTIONAL",
        "AND",
        "OR",
        "LPAREN",
        "RPAREN",
        "LBRACKET",
        "RBRACKET",
        "EQUAL",
    ]
    # tokens = ['REL_W_STR_ARG', 'NUMBER', 'VARNAME',]

    t_BLANK = r"__"

    REL_OPS = {
        "<": TregexMatcher.parent_of,
        ">": TregexMatcher.child_of,
        "<<": TregexMatcher.dominates,
        ">>": TregexMatcher.dominated_by,
        ">:": TregexMatcher.only_child_of,
        "<:": TregexMatcher.has_only_child,
        ">`": TregexMatcher.last_child_of_parent,
        ">-": TregexMatcher.last_child_of_parent,
        "<`": TregexMatcher.parent_of_last_child,
        "<-": TregexMatcher.parent_of_last_child,
        ">,": TregexMatcher.leftmost_child_of,
        "<,": TregexMatcher.has_leftmost_child,
        "<<`": TregexMatcher.has_rightmost_descendant,
        "<<-": TregexMatcher.has_rightmost_descendant,
        ">>`": TregexMatcher.rightmost_descendant_of,
        ">>-": TregexMatcher.rightmost_descendant_of,
        ">>,": TregexMatcher.leftmost_descendant_of,
        "<<,": TregexMatcher.has_leftmost_descendant,
        "$..": TregexMatcher.left_sister_of,
        "$++": TregexMatcher.left_sister_of,
        "$--": TregexMatcher.right_sister_of,
        "$,,": TregexMatcher.right_sister_of,
        "$.": TregexMatcher.immediate_left_sister_of,
        "$+": TregexMatcher.immediate_left_sister_of,
        "$-": TregexMatcher.immediate_right_sister_of,
        "$,": TregexMatcher.immediate_right_sister_of,
        "$": TregexMatcher.sister_of,
        "==": TregexMatcher.equals,
        "<=": TregexMatcher.parent_equals,
        "<<:": TregexMatcher.unary_path_ancestor_of,
        ">>:": TregexMatcher.unary_path_descedant_of,
        ":": TregexMatcher.pattern_splitter,
        ">#": TregexMatcher.immediately_heads,
        "<#": TregexMatcher.immediately_headed_by,
        ">>#": TregexMatcher.heads,
        "<<#": TregexMatcher.headed_by,
        # "..": TregexMatcher.precedes,
        # ",,": TregexMatcher.follows,
        # ".": TregexMatcher.immediately_precedes,
        # ",": TregexMatcher.immediately_follows,
    }
    # make sure long relations are checked first, or otherwise `>>` might
    # be tokenized as two `>`s.
    rels = sorted(REL_OPS.keys(), key=len, reverse=True)
    t_RELATION = "|".join(map(re.escape, rels))

    t_NEGATION = r"!"
    t_OPTIONAL = r"\?"
    t_AND = r"&"  # in `NP < NN | < NNS & > S`, `&` takes precedence over `|`
    t_OR = r"\|"
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_LBRACKET = r"\["
    t_RBRACKET = r"\]"
    t_EQUAL = r"="
    t_ignore = " \r\t"

    def t_ID(self, t):
        r"[^ 0-9\n\r(/|@!#&)=?[\]><~_.,$:{};][^ \n\r(/|@!#&)=?[\]><~.$:]*"
        # borrowed from Stanford Tregex:
        # https://github.com/stanfordnlp/CoreNLP/blob/main/src/edu/stanford/nlp/trees/tregex/TregexParser.jj#L66
        return t

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

    def __init__(self, tregex: str):
        self.lexer = lex.lex(module=self)
        self.lexer.input(tregex)
        self.backrefs_map: Dict[str, list] = {}
        # relation:str, modifier:Optional["!"|"?"]

    def findall(self, tree_string: str) -> tuple:
        trees = Tree.from_string(tree_string)
        parser = self.make_parser(trees)
        return parser.parse(lexer=self.lexer)

    def make_parser(self, trees: List[Tree]):
        tokens = self.tokens

        precedence = (
            ("left", "OR"),
            ("left", "RELATION", "AND"),
            ("nonassoc", "EQUAL"),
            ("left", "IMAGINE"),  # https://github.com/dabeaz/ply/issues/215
            # ("right", "AND"),
            # keep consistency with Stanford Tregex
            # 1. "VP < NP < N" matches a VP which dominates both an NP and an N
            # 2. "VP < (NP < N)" matches a VP dominating an NP, which in turn dominates an N
            # ("left", "RELATION"),
        )

        # 1. Label description
        # 1.1 simple label description
        def p_blank(p):
            """
            named_nodes : BLANK
            """
            p[0] = (list(TregexMatcher.match_any(trees)), None)  # name=None

        def p_id(p):
            """
            named_nodes : ID
            """
            label = p[1]
            p[0] = (list(TregexMatcher.match_label(trees, label, is_negate=False)), None)

        def p_regex(p):
            """
            named_nodes : REGEX
            """
            regex = p[1]
            p[0] = (
                list(node for node in TregexMatcher.match_regex(trees, regex, is_negate=False)),
                None,
            )

        # 1.2 node naming
        def p_equal_id(p):
            """
            named_nodes : named_nodes EQUAL ID
            """
            name = p[3]
            p[0] = (p[1][0], name)
            self.backrefs_map[name] = list(p[1][0])

        # 1.3 node negation
        def p_negation_id(p):
            """
            named_nodes : NEGATION ID
            """
            label = p[2]
            p[0] = (list(TregexMatcher.match_label(trees, label, is_negate=True)), None)

        def p_negation_regex(p):
            """
            named_nodes : NEGATION REGEX
            """
            regex = p[2]
            p[0] = (
                list(node for node in TregexMatcher.match_regex(trees, regex, is_negate=True)),
                None,
            )

        # 1.4 OR node description
        def p_or_nodes(p):
            """
            or_nodes : OR named_nodes
            """
            p[0] = p[2]

        def p_or_nodes_or_nodes(p):
            """
            or_nodes : or_nodes or_nodes
            """
            (these_nodes, this_name), (those_nodes, that_name) = p[1], p[2]
            p[0] = (these_nodes + those_nodes, that_name)
            if that_name is not None:
                self.backrefs_map[that_name] = these_nodes + those_nodes
            if this_name is not None:
                self.backrefs_map.pop(this_name)

        def p_nodes_or_nodes(p):
            """
            named_nodes : named_nodes or_nodes
            """
            (these_nodes, this_name), (those_nodes, that_name) = p[1], p[2]
            p[0] = (these_nodes + those_nodes, that_name)
            if that_name is not None:
                self.backrefs_map[that_name] = these_nodes + those_nodes
            if this_name is not None:
                self.backrefs_map.pop(this_name)

        # }}}
        # 2. Chain description
        def p_relation(p):
            """
            reduced_rel : RELATION
            """
            p[0] = (p[1], None)

        def p_negation_relation(p):
            """
            reduced_rel : NEGATION RELATION
            """
            p[0] = (p[2], "!")

        def p_optional_relation(p):
            """
            reduced_rel : OPTIONAL RELATION
            """
            p[0] = (p[2], "?")

        def p_reduced_rel_nodes(p):
            """
            and_conditions : reduced_rel named_nodes %prec IMAGINE
            """
            # %prec IMAGINE: https://github.com/dabeaz/ply/issues/215
            (rel, modifier), those_nodes = p[1:]
            p[0] = ((self.REL_OPS[rel], those_nodes, modifier),)

        def p_and_and_conditions(p):
            """
            and_conditions : AND and_conditions
            """
            p[0] = p[2]

        def p_and_conditions_and_conditions(p):
            """
            and_conditions : and_conditions and_conditions
            """
            p[0] = p[1] + p[2]

        def p_or_and_conditions(p):
            """
            or_conditions : OR and_conditions
            """
            and_conditions = p[2]
            p[0] = (and_conditions,)

        def p_or_conditions_or_conditions(p):
            """
            or_conditions : or_conditions or_conditions
            """
            p[0] = p[1] + p[2]

        def p_and_conditions(p):
            """
            chain : and_conditions
            """
            and_conditions, or_conditions = p[1], ()
            p[0] = (and_conditions, or_conditions)

        def p_and_conditions_or_conditions(p):
            """
            chain : and_conditions or_conditions
            """
            and_conditions, or_conditions = p[1], p[2]
            p[0] = (and_conditions, or_conditions)

        def p_lparen_chain_rparen(p):
            """
            chain : LPAREN chain RPAREN
            """
            p[0] = p[2]

        def p_lbracket_chain_rbracket(p):
            """
            chain : LBRACKET chain RBRACKET
            """
            p[0] = p[2]

        def p_nodes_chain(p):
            """
            named_nodes : named_nodes chain
            """
            these, (these_and_conditions, or_conditions) = p[1:]
            res, backrefs_map = TregexMatcher.and_(these, these_and_conditions)
            self.backrefs_map.update(backrefs_map)

            for those_and_conditions in or_conditions:
                nodes, backrefs_map = TregexMatcher.and_(these, those_and_conditions)
                res = (res[0] + nodes[0], res[1])

                this_name = these[1]
                for name in backrefs_map:
                    if name == this_name:
                        self.backrefs_map[this_name] += backrefs_map[name]
                    else:
                        self.backrefs_map[name] = backrefs_map[name]

            p[0] = res

        def p_lparen_nodes_rparen(p):
            """
            named_nodes : LPAREN named_nodes RPAREN
            """
            p[0] = p[2]

        def p_nodes(p):
            """
            pattern : named_nodes
            """
            p[0] = (p[1][0], self.backrefs_map)

        # def p_error(p):
        #     print(p)

        return yacc.yacc(debug=True, start="pattern")
