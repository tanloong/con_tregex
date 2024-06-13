#!/usr/bin/env python3

import contextlib
import sys


def stdout(s: str):
    with contextlib.suppress(BrokenPipeError):
        sys.stdout.write(s)
