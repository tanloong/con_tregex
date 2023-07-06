#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import argparse
import glob
import logging
import os
import sys
from typing import List, Optional, Tuple

from tregex import TregexPattern

# For all the procedures in TregexUI, return a tuple as the result The first
# element bool indicates whether the procedure succeeds The second element is
# the error message if it fails.
TregexProcedureResult = Tuple[bool, Optional[str]]

class TregexUI:
    def __init__(self) -> None:
        self.args_parser: argparse.ArgumentParser = self.create_args_parser()
        self.options: argparse.Namespace = argparse.Namespace()

    def create_args_parser(self) -> argparse.ArgumentParser:
        args_parser = argparse.ArgumentParser(
            prog="pytregex", formatter_class=argparse.RawDescriptionHelpFormatter
        )
        args_parser.add_argument("pattern")
        args_parser.add_argument(
            "--version",
            action="store_true",
            default=False,
            help="Show version and exit.",
        )
        return args_parser

    def parse_args(self, argv:List[str]) -> TregexProcedureResult:
        options, ipath_list = self.args_parser.parse_known_args(argv[1:])
        logging.basicConfig(format="%(message)s", level=logging.INFO)

        verified_ifile_list = []
        for path in ipath_list:
            if os.path.isfile(path):
                verified_ifile_list.append(path)
            elif os.path.isdir(path):
                verified_ifile_list.extend(glob.glob(f"{path}{os.path.sep}*.txt"))
            elif glob.glob(path):
                verified_ifile_list.extend(glob.glob(path))
            else:
                return (False, f"No such file as \n\n{path}")
        self.verified_ifile_list = verified_ifile_list

        self.options = options
        return True, None

    def run_matcher(self) -> TregexProcedureResult:
        if not self.verified_ifile_list:
            tree_string = "(VP (VP (VBZ Try) (NP (NP (DT this) (NN wine)) (CC and) (NP (DT these) (NNS snails)))) (PUNCT .))"
        else:
            tree_string = ""
            for ifile in self.verified_ifile_list:
                with open(ifile, 'r', encoding="utf-8") as f:
                    tree_string += f.read()

        pattern = TregexPattern(self.options.pattern)
        matches = pattern.findall(tree_string)
        for m in matches:
            sys.stdout.write(f"{m.to_string()}\n")

        return True, None
    def run(self) -> TregexProcedureResult:
        if self.options.version:
            return self.show_version()
        elif self.options.pattern:
            return self.run_matcher()
        else:
            self.args_parser.print_help()
            return True, None

    def show_version(self) -> TregexProcedureResult:
        from about import __version__
        print(__version__)
        return True, None

def main() -> None:
    ui = TregexUI()
    success, err_msg = ui.parse_args(sys.argv)
    if not success:
        logging.critical(err_msg)
        sys.exit(1)
    success, err_msg = ui.run()
    if not success:
        logging.critical(err_msg)
        sys.exit(1)
