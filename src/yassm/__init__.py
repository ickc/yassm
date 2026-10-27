"""yassm: Yet Another State Space Model package."""

from __future__ import annotations

import os

from .util import setup_logging

__version__ = "0.1.0"

# A library should not configure logging on import unless asked to:
# set YASSMLOGLEVEL (e.g. INFO) or call yassm.util.setup_logging() to see yassm's logs.
if os.environ.get("YASSMLOGLEVEL"):
    setup_logging()
