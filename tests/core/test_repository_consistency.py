"""Keep published examples and manuscript assets usable from a source checkout."""
import ast
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_sources_and_downloads(path):
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    examples = (ROOT / "docs/examples.rst").read_text()
    assert f"<../notebooks/{path.name}>" in examples
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source, filename=f"{path.name}:cell {index}")
        assert not any(output["output_type"] == "error" for output in cell["outputs"])
        # Literal HDF5 reads must ship their data alongside the notebook,
        # including a download in the website. Generated checkpoints use
        # write modes or the package's checkpoint API instead.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "h5py" and node.func.attr == "File"
                    and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == "r"):
                filename = node.args[0].value
                assert (path.parent / filename).is_file()
                assert f"<../notebooks/{filename}>" in examples


def test_manuscript_references_and_figure_inventory():
    paper = ROOT / "paper"
    source = (paper / "PyGFMC.tex").read_text()
    source += (paper / "table_rows.tex").read_text()
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    assert len(labels) == len(set(labels)), "duplicate manuscript labels"
    references = re.findall(r"\\(?:auto|eq|fig)?ref\{([^}]+)\}", source)
    assert not set(references) - set(labels), "undefined manuscript references"
    inventory = (paper / "README.md").read_text()
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    assert figures
    for figure in figures:
        path = paper / figure
        assert path.with_suffix(".pdf").is_file(), figure
        assert (path / "plot.py").is_file(), figure
        assert (path / "README.md").is_file(), figure
        assert f"`{figure}/`" in inventory, figure
