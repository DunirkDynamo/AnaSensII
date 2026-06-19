import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "AnaSens2"
author = "AnaSens Developers"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autodoc_typehints = "description"
html_theme = "furo"
