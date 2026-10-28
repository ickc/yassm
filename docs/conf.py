"""Sphinx configuration for the yassm documentation."""

from __future__ import annotations

import shutil
from pathlib import Path

import yassm

DOCS = Path(__file__).parent
ROOT = DOCS.parent

# The examples live in examples/; copy them next to the docs so that
# myst-nb can render (and execute) them in place with their data.
_examples_dst = DOCS / "examples"
shutil.rmtree(_examples_dst, ignore_errors=True)
shutil.copytree(ROOT / "examples", _examples_dst, ignore=shutil.ignore_patterns("__pycache__", "*.png"))

project = "yassm"
author = "Kolen Cheung"
copyright = f"2022–2026, {author}"
version = release = yassm.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = [
    "_build",
    "conf.py",
    "**/.ipynb_checkpoints",
    "examples/README.md",
    "examples/example_data.py",
    "examples/data/**",
]

# MyST / myst-nb ###############################################################

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
myst_heading_anchors = 3
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}
nb_execution_mode = "cache"
nb_execution_timeout = 1800
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True

# autodoc ######################################################################

autosummary_generate = True
autosummary_imported_members = False
# ignore-module-all: document each object once, in its own module, not again where it is re-exported (yassm.__all__)
autodoc_default_options = {"members": True, "show-inheritance": True, "ignore-module-all": True}
autodoc_member_order = "bysource"
# annotations are postponed (PEP 563) and refer to TYPE_CHECKING-only names:
# render them as written instead of resolving them
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
}

# HTML #########################################################################

html_theme = "furo"
html_title = f"yassm {version}"
html_theme_options = {
    "source_repository": "https://github.com/ickc/yassm",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ickc/yassm",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}
