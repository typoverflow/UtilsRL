# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
# # sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

# import UtilsRL
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'UtilsRL'
copyright = '2022, typoverflow'
author = 'typoverflow'
release = (ROOT_DIR / "VERSION").read_text()



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme", 
    "sphinx.ext.autodoc", 
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = [".rst"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Auto API options
# autoapi_type = 'python'
# autoapi_dirs = ["../../UtilsRL"]
# autoapi_options = [ 'members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'special-members']

# autodoc options
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# autodoc_default_options = {
#     "special-members":
#     ", ".join(
#         [
#             # "__len__",
#             # "__call__",
#             # "__getitem__",
#             # "__setitem__",
#             # "__getattr__",
#             # "__setattr__",
#         ]
#     )
# }
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_member_order = "bysource"


html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]
def setup(app):
    app.add_css_file("css/style.css")