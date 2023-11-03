#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import argparse
import glob
import logging
import os
import sys
from typing import List, Optional, Tuple

from pytregex.tregex import TregexPattern

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
            prog="pytregex",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False,
        )
        args_parser.add_argument("--help", action="help", help="Show this help message and exit")
        args_parser.add_argument("pattern", help="Tregex pattern")
        args_parser.add_argument(
            "-filter",
            action="store_true",
            dest="is_stdin",
            default=False,
            help="read tree input from stdin",
        )
        args_parser.add_argument(
            "-h",
            metavar="<handle>",
            action="extend",
            nargs="+",
            dest="handles",
            help=(
                "for each node-handle specified, the node matched and given that handle will be"
                " printed. Multiple nodes can be printed by using this option multiple times on"
                " a single command line."
            ),
        )
        args_parser.add_argument(
            "--version",
            action="store_true",
            default=False,
            help="Show version and exit.",
        )
        args_parser.add_argument(
            "--verbose",
            action="store_true",
            dest="is_verbose",
            default=False,
            help="Give verbose debugging output.",
        )
        args_parser.add_argument(
            "--quiet",
            action="store_true",
            dest="is_quiet",
            default=False,
            help="Stop pytregex from print anything.",
        )
        return args_parser

    def parse_args(self, argv: List[str]) -> TregexProcedureResult:
        idx: Optional[int] = None
        if "--" in argv[1:]:
            idx = argv.index("--")
        if idx is not None:
            options, ipath_list = self.args_parser.parse_args(argv[1:idx]), argv[idx + 1 :]
        else:
            options, ipath_list = self.args_parser.parse_known_args(argv[1:])

        if options.is_verbose and options.is_quiet:
            return False, "--verbose and --quiet cannot be set at the same time"
        if options.is_verbose:
            logging_level = logging.DEBUG
        elif options.is_quiet:
            logging_level = logging.WARNING
        else:
            logging_level = logging.INFO
        logging.basicConfig(format="%(message)s", level=logging_level)

        self.tree_string = None
        self.verified_ifile_list = None
        if options.is_stdin:
            if ipath_list:
                return (
                    False,
                    "Input files are unaccepted when reading tree input from stdin: \n\n{}"
                    .format("\n".join(ipath_list)),
                )
            self.tree_string = sys.stdin.read()
        else:
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
            if verified_ifile_list:
                self.verified_ifile_list = verified_ifile_list

        self.options = options
        return True, None

    def run_matcher(self) -> TregexProcedureResult:
        default_tree_string = (
            "(VP (VP (VBZ Try) (NP (NP (DT this) (NN wine)) (CC and) (NP (DT these) (NNS"
            " snails)))) (PUNCT .))"
        )

        if self.verified_ifile_list is not None:
            tree_string = ""
            for ifile in self.verified_ifile_list:
                logging.debug(f"Reading tree input from input {ifile}...")
                with open(ifile, "r", encoding="utf-8") as f:
                    tree_string += f.read()
        elif self.tree_string is not None:
            logging.debug("Reading tree input from stdin...")
            tree_string = self.tree_string
        else:
            logging.debug(f"No tree input. Using the default {default_tree_string}.")
            tree_string = default_tree_string

        pattern = TregexPattern(self.options.pattern)
        matches = pattern.findall(tree_string)

        if self.options.handles:
            logging.debug("Printing handles...")
            for handle in self.options.handles:
                handled_nodes = pattern.get_nodes(handle)
                for node in handled_nodes:
                    sys.stdout.write(f"{node}\n")
        else:
            logging.debug("Printing matches...")
            for m in matches:
                sys.stdout.write(f"{m}\n")
        logging.info(f"There were {len(matches)} matches in total.")

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
