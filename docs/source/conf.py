import sourcefinder

project = "PySE"
authors = [
    "Hanno Spreeuw",
    "John Swinbank",
    "Gijs Molenaar",
    "Tim Staley",
    "Evert Rol",
    "John Sanders",
    "Bart Scheers",
    "Mark Kuiack",
    "Suvayu Ali",
    "Timo Millenaar",
    "Antonia Rowlinson",
]
author = ", ".join(authors)
copyright = ", ".join(["2025", *authors])
release = sourcefinder.__version__
version = release.split(".dev")[0] if ".dev" in release else release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "numpydoc",
    "autoapi.extension",
]

autodoc_typehints = "both"

# autoapi_keep_files = True
autoapi_dirs = ["../../sourcefinder"]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_member_order = "groupwise"

templates_path = ["_templates"]

# source file discovery
exclude_patterns = []

# HTML options
html_theme = "pydata_sphinx_theme"
