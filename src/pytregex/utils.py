#!/usr/bin/env python3

from typing import Optional, Tuple

# For all the procedures in TregexUI, return a tuple as the result The first
# element bool indicates whether the procedure succeeds The second element is
# the error message if it fails.
TregexProcedureResult = Tuple[bool, Optional[str]]
