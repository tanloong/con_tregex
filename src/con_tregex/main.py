#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from tregex import TregexPattern


def main():
    tree_string = """
(ROOT
  (S
    (NP (EX There))
    (VP (VBD was)
      (NP
        (NP (DT no) (NNS possibility))
        (PP (IN of)
          (S
            (VP (VBG taking)
              (NP (DT a) (NN walk))
              (NP (DT that) (NN day)))))))
    (. .)))
"""

    # data = "NP=n1 < NN=n2 | < NNS=n3 > S=n4"{{{
    # data = "NP=n3 < (NN=n2 < day=n1)"
    # data = "(VP|NN|day)=y"
    # data = "VP < NP"
    # data = "NP !< NN"
    # data = "NN=x ?>> NP"
    # data = "NP=x [<< NN | << NNS] >> S"
    # data = "NP << [NN << NNS]"
    # data = "NP << NN | << NNS & > S"
    # data = "NP=x << NN=y | << NNS=z"
    # data = "NP << NNS=z"
    # data = "NP | NN"
    # data = "NP <` NN"}}}
    data = "__ <<# IN"

    tregex1 = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, __=m <- (__ <, CONJ <- __=z))))")
    tregex2 = TregexPattern("PNT=p >>- (/(.+)/#1%var=l >, (__=t <- (__=r <, /(.+)/#1%var=m <- (__ <, CONJ <- /(.+)/#1%var=z))))")
    tregex3 = TregexPattern("PNT=p >>- (__=l >, (__=t <- (__=r <, ~l <- (__ <, CONJ <- ~l))))")
    tree_string = "(T (X (N (N Moe (PNT ,)))) (NP (X (N Curly)) (NP (CONJ and) (X (N Larry)))))"

    # pattern = TregexPattern(data)
    # while True:{{{
    #     tok = pattern.lexer.token()
    #     if tok is None:
    #         break
    #     print(tok)
    # breakpoint()

    # x = pattern.findall(tree)
    # breakpoint()
    # }}}
    matches = tregex1.findall(tree_string)
    # t = matches[0]
    for match in matches:
        print(match)
    print("=" * 60)
    for name, nodes in tregex1.backrefs_map.items():
        for node in nodes:
            print(f"{name}: {node}")
