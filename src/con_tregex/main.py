#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from .tregex import TregexPattern


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
    # (ROOT
    #   (S
    #     (NP (EX There))
    #     (VP (VBD was)
    #       (NP
    #         (NP (DT no) (NNS possibility))
    #         (PP (IN of)
    #           (S
    #             (VP (VBG taking)
    #               (NP (DT a) (NN walk))
    #               (NP (DT that) (NN day)))))))
    #     (. .)))
    ##

    # data = "NP=n1 < NN=n2 | < NNS=n3 > S=n4"
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
    # data = "NP <` NN"
    data = "__ <<# IN"
    pattern = TregexPattern(data)
    # while True:
    #     tok = pattern.lexer.token()
    #     if tok is None:
    #         break
    #     print(tok)
    # breakpoint()

    # x = pattern.findall(tree)
    # breakpoint()

    matches, backrefs_map = pattern.findall(tree_string)
    t = matches[0]
    for match in matches:
        print(match)
    print("=" * 60)
    for name, nodes in backrefs_map.items():
        for node in nodes:
            print(f"{name}: {node}")
