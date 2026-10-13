# Sphinx configuration for PyGFMC documentation.
import os
import sys

# Make the package in ../codes importable for the autodoc directives of api.rst.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'codes')))

# -- Project ----------------------------------------------------------------
project   = 'PyGFMC'
copyright = '2026, Robin Schaefer'
author    = 'Robin Schaefer'
import pygfmc                       # noqa: E402  (path set above)
release   = pygfmc.__version__      # the one place the number is written: codes/pyproject.toml
version   = release

# -- General ----------------------------------------------------------------
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',        # API reference from the docstrings
    'sphinx.ext.autosummary',    # summary table at the top of each section
    'sphinx.ext.napoleon',       # Google-style Args:/Returns: docstrings
    'nbsphinx',                  # render Jupyter notebooks
]

nbsphinx_execute = 'never'       # don't re-run notebooks during build

exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- autodoc ---------------------------------------------------------------
autodoc_default_options = {
    'members': True,               # documented public methods of classes
    'member-order': 'bysource',
    'exclude-members': 'tree_flatten, tree_unflatten',
}
autodoc_typehints   = 'description'
autodoc_mock_imports = ['netket', 'optax', 'flax']   # optional extras
napoleon_google_docstring = True
napoleon_numpy_docstring  = False
autosummary_generate = False       # tables only; the entries are written in api.rst

# -- HTML -------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'   # pip install sphinx_rtd_theme
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 2,
}
html_static_path = ['_static']
html_css_files   = ['custom.css']
# review markup: :red:`new text`, :del:`removed text` (red, struck through) and
# `.. container:: red` for whole new paragraphs; see _static/custom.css
rst_prolog = '''
.. role:: red
.. role:: del
'''
# copy the paper into the site root, so paper.rst can embed it as src="PyGFMC.pdf"
html_extra_path = ['../paper/PyGFMC.pdf']

# -- intersphinx -----------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax':    ('https://jax.readthedocs.io/en/latest', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
}
intersphinx_timeout = 10           # do not hang the build when offline

# The Hamiltonian classes are NamedTuples whose generated constructor takes the
# internal, JIT-ready fields (edge_i, J_pm, has_jz, ...).  Users build them with
# the `create` classmethod, so hide the generated signature for those classes;
# `create` is documented as the first member.
autodoc_typehints_description_target = 'documented'
autodoc_preserve_defaults = True     # show defaults as in the source (jnp.float32)


def _hide_dataclass_signature(app, what, name, obj, options, signature, return_annotation):
    if what == 'class' and name.startswith('pygfmc.hamiltonian.') and hasattr(obj, 'create'):
        return '', None
    # The protocol classes are interfaces, never instantiated: no signature.
    if what == 'class' and name.startswith('pygfmc.protocols.'):
        return '', None
    return signature, return_annotation


def setup(app):
    app.connect('autodoc-process-signature', _hide_dataclass_signature)
