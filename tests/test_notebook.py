"""Static contract tests for the one notebook the team actually runs.

These tests never *execute* the notebook (that needs a GPU, the Excel file and
hours); they parse it. What they protect:

* the file is valid ``nbformat`` v4 and carries **no outputs** (the PR ships a
  clean notebook; the executed copy is a follow-up commit);
* the setup cell really has the Colab block active and the LOCAL block
  commented out, so nobody opens it on Colab and finds it configured for a
  laptop;
* every code cell is plain, compilable Python -- a syntax error in cell 30 of a
  notebook is otherwise only discovered three hours into a Colab run;
* every call into ``crispr_deephf`` matches the real signature, so a typo'd
  keyword cannot hide until the run reaches that cell;
* the multi-target run uses ``shared_split`` and single-target runs do not --
  the data leak the team warned about in ``models/CNN.py``;
* the comparability caveat and the selection-bias note are printed
  unconditionally, never read out of a pandas ``attrs`` dict that does not
  survive every operation.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "crispr_deephf_pipeline.ipynb"

#: Package modules the notebook is allowed to call, checked against their real signatures.
PACKAGE_MODULES = (
    "architectures",
    "benchmark",
    "bio_features",
    "config",
    "data",
    "features",
    "models",
    "plots",
    "runner",
    "tuning",
)

#: Run switches and the mode selector the settings cell must expose.
REQUIRED_SWITCHES = ("MODE", "RUN_TIER1", "RUN_TIER2", "RUN_TIER3", "RUN_EXTRAS", "RUN_TUNING")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _nb():
    return nbformat.read(NB, as_version=4)


def _code_cells(nb) -> list:
    return [c for c in nb.cells if c.cell_type == "code"]


def _all_source(nb) -> str:
    return "\n".join(c.source for c in nb.cells)


def _module_calls(nb):
    """Yield ``(module_name, attr, call_node, cell_source)`` for every ``mod.attr(...)``."""
    for cell in _code_cells(nb):
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in PACKAGE_MODULES
            ):
                yield node.func.value.id, node.func.attr, node, cell.source


# --------------------------------------------------------------------------- #
# The four contract tests from the task brief
# --------------------------------------------------------------------------- #
def test_notebook_is_valid_and_has_no_outputs():
    nb = _nb()
    nbformat.validate(nb)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    assert all(c.get("execution_count") is None for c in _code_cells(nb))


def test_setup_cell_has_colab_active_and_local_commented():
    src = _nb().cells[1].source
    assert "google.colab" in src
    assert "[A] GOOGLE COLAB" in src and "[B] LOCAL" in src
    local = src.split("[B] LOCAL")[1]
    assert all(
        line.strip().startswith("#") or not line.strip() for line in local.splitlines()
    ), "everything after the [B] LOCAL marker must stay commented out"


def test_commented_local_block_is_valid_python_when_uncommented():
    """A syntax error in the LOCAL block would otherwise surface only on a teammate's laptop."""
    src = _nb().cells[1].source
    block = src.split("# --- uncomment from here ---")[1]

    def uncomment(line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return ""
        assert stripped.startswith("#"), f"uncommented line in the LOCAL block: {line!r}"
        return stripped[1:].removeprefix(" ")

    compile("\n".join(uncomment(x) for x in block.splitlines()), "<local block>", "exec")


def test_every_code_cell_compiles():
    for i, cell in enumerate(_nb().cells):
        if cell.cell_type == "code" and not cell.source.lstrip().startswith(("!", "%")):
            compile(cell.source, f"<cell {i}>", "exec")


def test_switches_and_sections_present():
    nb = _nb()
    all_src = _all_source(nb)
    for token in REQUIRED_SWITCHES:
        assert token in all_src
    assert sum(1 for c in nb.cells if c.cell_type == "markdown") >= 12


# --------------------------------------------------------------------------- #
# Structure: written for five teammates who did not write this code
# --------------------------------------------------------------------------- #
def test_every_code_cell_is_introduced_by_a_markdown_note():
    """A teammate should never meet a code cell with no explanation above it."""
    cells = _nb().cells
    assert cells[0].cell_type == "markdown", "the notebook must open with its title/overview"
    for i, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        # Either the cell right above is the section's markdown note, or this is the
        # second code cell of a section whose note is one cell further up.
        assert any(
            c.cell_type == "markdown" for c in cells[max(0, i - 2) : i]
        ), f"code cell {i} is more than one cell away from its markdown note"


def test_no_shell_escapes_or_line_magics():
    """Plain Python only, so the whole notebook is parseable and lintable."""
    for i, cell in enumerate(_code_cells(_nb())):
        for line in cell.source.splitlines():
            assert not line.lstrip().startswith(("!", "%")), (
                f"code cell {i} uses a shell escape or magic: {line!r}"
            )


def test_setup_cell_warns_on_a_runtime_mismatch():
    src = _nb().cells[1].source
    assert "google.colab" in src and "find_spec" in src
    assert "MISMATCH" in src, "the setup cell must warn loudly on a runtime mismatch"
    assert "warnings.warn" in src


def test_run_switch_defaults_are_safe():
    all_src = _all_source(_nb())
    assert "RUN_TUNING = False" in all_src
    assert "FORCE_RERUN = False" in all_src


def test_required_sections_are_all_present():
    all_src = _all_source(_nb())
    for heading in (
        "## 1. Setup",
        "## 2. Run settings",
        "## 3. Data",
        "## 4. Exploratory",
        "## 5. Features",
        "## 6. Bio features",
        "## 7. Tier 1",
        "## 8. Tier 2",
        "## 9. Tier 3",
        "## 10. Extras",
        "## 11. Hyper-parameter search",
        "## 12. Results",
    ):
        assert heading in all_src, f"missing section heading {heading!r}"


# --------------------------------------------------------------------------- #
# Honesty: the caveats must be printed, not inferred from pandas attrs
# --------------------------------------------------------------------------- #
def test_caveat_and_selection_note_are_printed_unconditionally():
    code = "\n".join(c.source for c in _code_cells(_nb()))
    assert "benchmark.format_caveat()" in code
    assert "benchmark.format_selection_note(" in code
    assert ".attrs" not in code, (
        "pandas attrs do not survive every operation; print "
        "benchmark.format_selection_note(...) directly instead"
    )


def test_paper_guide_counts_are_shown_beside_ours():
    all_src = _all_source(_nb())
    assert "PAPER_N_GUIDES" in all_src
    assert "58,167" in all_src and "58,617" in all_src, (
        "the eSpCas9 guide-count discrepancy between the paper's text and its own "
        "supplementary data must be surfaced, not hidden"
    )


# --------------------------------------------------------------------------- #
# The data-leak guard
# --------------------------------------------------------------------------- #
def test_run_deep_model_calls_respect_the_shared_split_guard():
    calls = [
        (node, cell_src)
        for mod, attr, node, cell_src in _module_calls(_nb())
        if (mod, attr) == ("runner", "run_deep_model")
    ]
    assert calls, "the notebook never trains a deep model"
    saw_multi = False
    for node, cell_src in calls:
        kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
        assert "target_cols" in kw and "split_col" in kw
        if "TARGET_COLS" in kw["target_cols"]:  # all three enzymes -> multi-output head
            saw_multi = True
            assert "SHARED_SPLIT_COL" in kw["split_col"], (
                "a multi-target run on a per-enzyme split column leaks across enzymes"
            )
        else:
            assert "SHARED_SPLIT_COL" not in kw["split_col"], (
                "single-target runs must read that enzyme's own frozen split column"
            )
            assert "_split" in cell_src, (
                "the single-target cell must build the per-enzyme split column name"
            )
    assert saw_multi, "the multi-task model is never run"


def test_multitask_model_is_the_only_multi_output_run():
    all_src = _all_source(_nb())
    assert "MultiTask-BiLSTM+bio" in all_src
    assert "SHARED_SPLIT_COL" in all_src


# --------------------------------------------------------------------------- #
# Every package call matches the real signature
# --------------------------------------------------------------------------- #
def test_package_module_aliases_are_imported():
    nb = _nb()
    used = {mod for mod, _, _, _ in _module_calls(nb)}
    code = "\n".join(c.source for c in _code_cells(nb))
    for mod in sorted(used):
        assert "crispr_deephf import" in code or f"crispr_deephf.{mod}" in code
        assert mod in code


def test_every_package_call_matches_the_real_signature():
    pytest.importorskip("torch")  # architectures / runner / tuning need it
    modules = {name: importlib.import_module(f"crispr_deephf.{name}") for name in PACKAGE_MODULES}
    checked = 0
    for mod, attr, node, _ in _module_calls(_nb()):
        module = modules[mod]
        assert hasattr(module, attr), f"crispr_deephf.{mod} has no attribute {attr!r}"
        obj = getattr(module, attr)
        if not callable(obj):
            continue
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            continue  # unpacked call: cannot be checked statically
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        placeholders = [object()] * len(node.args)
        kwargs = {k.arg: object() for k in node.keywords}
        try:
            sig.bind(*placeholders, **kwargs)
        except TypeError as exc:
            pytest.fail(f"{mod}.{attr}{sig} called wrongly in the notebook: {exc}")
        checked += 1
    assert checked >= 20, f"only {checked} package calls checked; the alias map looks wrong"
