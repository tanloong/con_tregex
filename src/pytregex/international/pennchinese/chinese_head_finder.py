#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from pytregex.abstract_collins_head_finder import AbstractCollinsHeadFinder

# translated from https://github.com/stanfordnlp/CoreNLP/blob/main/src/edu/stanford/nlp/trees/international/pennchinese/ChineseHeadFinder.java
# last modified at Apr 3, 2017 (https://github.com/stanfordnlp/CoreNLP/commits/main/src/edu/stanford/nlp/trees/international/pennchinese/ChineseHeadFinder.java)


class ChineseHeadFinder(AbstractCollinsHeadFinder):
    def __init__(self) -> None:
        leftExceptPunct = ["leftexcept", "PU"]
        rightExceptPunct = ["rightexcept", "PU"]

        coordSwitch = False
        left = "left" if not coordSwitch else "right"
        right = "right" if not coordSwitch else "left"

        rightdis = "rightdis"
        self.defaultRule = [right]

        self.nonTerminalInfo = {
            # ROOT is not always unary for chinese -- PAIR is a special notation
            # that the Irish people use for non-unary ones....
            "ROOT": [[left, "IP"]],
            "PAIR": [[left, "IP"]],
            # Major syntactic categories
            "ADJP": [
                [left, "JJ", "ADJP"]
            ],  # there is one ADJP unary rewrite to AD but otherwise all have JJ or ADJP
            "ADVP": [
                [left, "AD", "CS", "ADVP", "JJ"]
            ],  # CS is a subordinating conjunctor, and there are a couple of ADVP->JJ unary rewrites
            "CLP": [[right, "M", "CLP"]],
            # "CP", [[left, "WHNP","IP","CP","VP"]], # this is complicated; see bracketing guide p. 34.  Actually, all WHNP are empty.  IP/CP seems to be the best semantic head; syntax would dictate DEC/ADVP. Using IP/CP/VP/M is INCREDIBLY bad for Dep parser - lose 3% absolute.
            "CP": [
                [right, "DEC", "WHNP", "WHPP"],
                rightExceptPunct,
            ],  # the (syntax-oriented) right-first head rule
            # "CP": [[right, "DEC", "ADVP", "CP", "IP", "VP", "M"]], # the (syntax-oriented) right-first head rule
            "DNP": [
                [right, "DEG", "DEC"],
                rightExceptPunct,
            ],  # according to tgrep2, first preparation, all DNPs have a DEG daughter
            "DP": [[left, "DT", "DP"]],  # there's one instance of DP adjunction
            "DVP": [[right, "DEV", "DEC"]],  # DVP always has DEV under it
            "FRAG": [
                [right, "VV", "NN"],
                rightExceptPunct,
            ],  # FRAG seems only to be used for bits at the beginnings of articles: "Xinwenshe<DATE>" and "(wan)"
            "IP": [
                [left, "VP", "IP"],
                rightExceptPunct,
            ],  # CDM July 2010 following email from Pi-Chuan changed preference to VP over IP: IP can be -SBJ, -OBJ, or -ADV, and shouldn't be head
            "LCP": [[right, "LC", "LCP"]],  # there's a bit of LCP adjunction
            "LST": [[right, "CD", "PU"]],  # covers all examples
            "NP": [
                [right, "NN", "NR", "NT", "NP", "PN", "CP"]
            ],  # Basic heads are NN/NR/NT/NP; PN is pronoun.  Some NPs are nominalized relative clauses without overt nominal material; these are NP->CP unary rewrites.  Finally, note that this doesn't give any special treatment of coordination.
            "PP": [
                [left, "P", "PP"]
            ],  # in the manual there's an example of VV heading PP but I couldn't find such an example with tgrep2
            # cdm 2006: PRN changed to not choose punctuation.  Helped parsing (if not significantly)
            # "PRN": [[left, "PU"]], #presumably left/right doesn't matter
            "PRN": [
                [left, "NP", "VP", "IP", "QP", "PP", "ADJP", "CLP", "LCP"],
                [rightdis, "NN", "NR", "NT", "FW"],
            ],
            # cdm 2006: QP: add OD -- occurs some; occasionally NP, NT, M; parsing performance no-op
            "QP": [
                [right, "QP", "CLP", "CD", "OD", "NP", "NT", "M"]
            ],  # there's some QP adjunction
            # add OD?
            "UCP": [
                [
                    left,
                ]
            ],  # an alternative would be "PU","CC"
            "VP": [
                [
                    left,
                    "VP",
                    "VCD",
                    "VPT",
                    "VV",
                    "VCP",
                    "VA",
                    "VC",
                    "VE",
                    "IP",
                    "VSB",
                    "VCP",
                    "VRD",
                    "VNV",
                ],
                leftExceptPunct,
            ],  # note that ba and long bei introduce IP-OBJ small clauses; short bei introduces VP
            # add BA, LB, as needed
            # verb compounds
            "VCD": [[left, "VCD", "VV", "VA", "VC", "VE"]],  # could easily be right instead
            "VCP": [[left, "VCD", "VV", "VA", "VC", "VE"]],  # not much info from documentation
            "VRD": [[left, "VCD", "VRD", "VV", "VA", "VC", "VE"]],  # definitely left
            "VSB": [
                [right, "VCD", "VSB", "VV", "VA", "VC", "VE"]
            ],  # definitely right, though some examples look questionably classified (na2lai2 zhi1fu4)
            "VNV": [[left, "VV", "VA", "VC", "VE"]],  # left/right doesn't matter
            "VPT": [[left, "VV", "VA", "VC", "VE"]],  # activity verb is to the left
            # some POS tags apparently sit where phrases are supposed to be
            "CD": [[right, "CD"]],
            "NN": [[right, "NN"]],
            "NR": [[right, "NR"]],
            # I'm adding these POS tags to do primitive morphology for character-level
            # parsing.  It shouldn't affect anything else because heads of preterminals are not
            # generally queried - GMA
            "VV": [[left]],
            "VA": [[left]],
            "VC": [[left]],
            "VE": [[left]],
            # new for ctb6.
            "FLR": [rightExceptPunct],
            # new for CTB9
            "DFL": [rightExceptPunct],
            "EMO": [leftExceptPunct],  # left/right doesn't matter
            "INC": [leftExceptPunct],
            "INTJ": [leftExceptPunct],
            # old version suitable for v5.1 ... does not cover "我的天 哪" for example
            # "INTJ": [[right, "INTJ", "IJ", "SP"]],
            "OTH": [leftExceptPunct],
            "SKIP": [leftExceptPunct],
        }
