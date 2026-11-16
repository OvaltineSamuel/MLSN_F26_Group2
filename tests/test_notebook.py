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
  keyword cannot hide until the run reaches that cell. The package modules are
  *parsed*, not imported, so this check runs in CI as well -- CI installs the dev
  extra only, and ``runner`` / ``architectures`` / ``tuning`` cannot be imported
  without torch. One separate test does import them, purely to confirm the
  AST-rebuilt signatures agree with what Python itself reports; only that one
  test skips without torch;
* the multi-target run uses ``shared_split`` and single-target runs do not --
  the data leak the team warned about in ``models/CNN.py``;
* the comparability caveat and the selection-bias note are printed
  unconditionally, never read out of a pandas ``attrs`` dict that does not
  survive every operation.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[1]
#: The notebook under contract: the clean, unexecuted copy the team edits and runs.
NB = ROOT / "notebooks" / "crispr_deephf_pipeline.ipynb"
#: Where an **executed** copy goes, outputs and all, after a ``MODE = "full"`` run.
#:
#: Two tests here would otherwise turn CI red the moment that copy is committed:
#: :func:`test_notebook_is_valid_and_has_no_outputs` (an executed notebook is nothing but
#: outputs) and :func:`test_run_switch_defaults_are_safe` (which forbids ``MODE = "full"``
#: in any code cell -- the reported run requires it). Both read :data:`NB` only, so the
#: results commit lands under this directory and is not held to the shipping defaults.
#: The convention is documented in the README so it is not folklore.
EXECUTED_DIR = ROOT / "notebooks" / "executed"
SRC = ROOT / "src" / "crispr_deephf"

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

#: The teammates' own notebooks, kept byte-for-byte as they wrote them, outputs and all.
_TEAMMATE_NOTEBOOKS = frozenset({"BiLSTM.ipynb", "CNN_Train.ipynb"})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _nb():
    return nbformat.read(NB, as_version=4)


def _code_cells(nb) -> list:
    return [c for c in nb.cells if c.cell_type == "code"]


def _all_source(nb) -> str:
    return "\n".join(c.source for c in nb.cells)


def _notebook_assignments(nb) -> dict[str, object]:
    """Every literal top-level ``NAME = <literal>`` in the code cells, later cells winning.

    Substring checks on the cell text cannot tell an assignment from the same words inside a
    printed message, which is exactly where ``RUN_TIER3 = True`` now appears (in the
    ViennaRNA warning), so the switches are read from the AST instead.
    """
    out: dict[str, object] = {}
    for cell in _code_cells(nb):
        for node in ast.parse(cell.source).body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    continue  # not a literal (a call, a comprehension, ...)
    return out


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
    """Scoped to the unexecuted contract notebook; see :data:`EXECUTED_DIR`."""
    assert EXECUTED_DIR not in NB.parents, "the contract notebook is the unexecuted one"
    nb = _nb()
    nbformat.validate(nb)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    assert all(c.get("execution_count") is None for c in _code_cells(nb))


def test_the_executed_copy_has_a_home_of_its_own_and_is_documented():
    """The follow-up commit of the executed notebook must not be able to fail CI.

    Decided up front rather than discovered when the results commit goes red: the executed
    copy lives in ``notebooks/executed/``, the two defaults-and-outputs contracts above read
    only :data:`NB`, and the README states the convention. Any executed notebook found
    *outside* that directory is the mistake this test exists to catch.
    """
    assert "notebooks/executed/" in (ROOT / "README.md").read_text(encoding="utf-8"), (
        "the executed-notebook convention must be documented in the README"
    )
    strays = []
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        if path == NB or path.name in _TEAMMATE_NOTEBOOKS:
            continue
        nb = nbformat.read(path, as_version=4)
        if any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
            strays.append(path.name)
    assert strays == [], (
        f"executed notebooks belong in notebooks/executed/, not beside the pipeline: {strays}"
    )


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
    """The shipped defaults must not start hours of GPU work on the first `Run all`.

    Scoped to the unexecuted contract notebook (:data:`NB`) on purpose -- see
    :data:`EXECUTED_DIR`: the executed copy of a ``MODE = "full"`` run lands under
    ``notebooks/executed/`` and is not held to this.

    ``MODE`` and ``RUN_TUNING`` were not the whole cost. ``RUN_TIER3`` / ``RUN_EXTRAS``
    shipped ``True``, and quick mode does not make them cheap: ``run_deep_model`` folds the
    enzyme's **full** row set whatever ``MODE`` says, so a quick Tier-3 + extras run folded
    230,961 sequences (~2.5 h at the measured 38.7 ms/guide) before the first reported epoch
    -- in a notebook that tells the reader quick mode "runs end to end in minutes".
    """
    code = "\n".join(c.source for c in _code_cells(_nb()))
    assert 'MODE = "quick"' in code, (
        'MODE must ship as "quick" (3 epochs, 1 seed); flipping to "full" is a deliberate edit'
    )
    assert 'MODE = "full"' not in code, (
        'no code cell may set MODE = "full"; the markdown tells the reader to change it'
    )
    # Read the assignments themselves: the strings "RUN_TIER3 = True" and the like also
    # occur inside the cells' own printed warning messages.
    assigned = _notebook_assignments(_nb())
    assert assigned["MODE"] == "quick"
    for switch in ("RUN_TIER3", "RUN_EXTRAS", "RUN_TUNING", "FORCE_RERUN"):
        assert assigned[switch] is False, (
            f"{switch} must ship as False: it costs hours before any reported number"
        )
    # Tier 1 and Tier 2 are sequence-only and cheap even in full mode, so they stay on.
    assert assigned["RUN_TIER1"] is True and assigned["RUN_TIER2"] is True


def test_the_tiers_that_need_bio_features_are_gated_on_viennarna():
    """Cell 21 computes ``VIENNA_AVAILABLE``; the tiers that cannot run without it must read it.

    Every Tier-3 and extras model takes the bio branch, so a failed ``[deep]`` install should
    skip them with one clear line rather than fail per model deep inside a run.
    """
    code = "\n".join(c.source for c in _code_cells(_nb()))
    for switch in ("RUN_TIER3", "RUN_EXTRAS"):
        assert f"if {switch} and bio_features.VIENNA_AVAILABLE:" in code, (
            f"{switch} must also require bio_features.VIENNA_AVAILABLE"
        )
    assert code.count("if RUN_EXTRAS and bio_features.VIENNA_AVAILABLE:") >= 2, (
        "both the single-target extras and the multi-task model need the gate"
    )


def test_colab_block_clones_this_branch_and_warns_on_a_stale_clone():
    """A bare ``git clone <url>`` takes the default branch, which pre-merge is ``main``.

    ``main`` has no ``crispr_deephf`` package, so a reader who clicks the README badge before
    this branch merges would get a ModuleNotFoundError with no hint that the branch was why.
    """
    src = _nb().cells[1].source
    assert "REPO_BRANCH" in src
    assert '"clone", "-b", REPO_BRANCH' in src, "the clone must name the branch explicitly"
    pull = src.split("if REPO_DIR.exists():")[1].split("else:")[0]
    assert "returncode" in pull and "warnings.warn" in pull, (
        "a failed `git pull --ff-only` leaves stale code behind and must say so, not pass "
        "in silence"
    )


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
    assert "benchmark.format_caveat(" in code
    assert "benchmark.format_selection_note(" in code
    assert ".attrs" not in code, (
        "pandas attrs do not survive every operation; print "
        "benchmark.format_selection_note(...) directly instead"
    )


def test_the_caveat_is_told_how_many_seeds_this_run_has():
    """``config.PAPER_CAVEAT`` no longer claims a seed count, so the notebook must supply it.

    It used to read "averaged over 3 model-init seeds" whatever ran -- a method quick mode
    never used, printed right beside a table whose s.d. column said ``n/a``.
    """
    code = "\n".join(c.source for c in _code_cells(_nb()))
    assert "N_SEEDS_THIS_RUN" in code
    assert 'results_long["Seed"].nunique()' in code, (
        "the seed count must be counted from the results on disk, not written by hand"
    )
    assert "benchmark.format_caveat()" not in code, (
        "every printed caveat must pass the real seed count: format_caveat(N_SEEDS_THIS_RUN)"
    )
    assert code.count("benchmark.format_caveat(N_SEEDS_THIS_RUN)") >= 3


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
        # every enzyme in play (ACTIVE_TARGETS, or all of TARGET_COLS) -> multi-output head
        if "TARGET_COLS" in kw["target_cols"] or "ACTIVE_TARGETS" in kw["target_cols"]:
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
def _imported_package_aliases(nb) -> set[str]:
    """Names actually bound by an import of ``crispr_deephf`` in the notebook's code cells.

    Handles ``from crispr_deephf import runner [as r]`` and ``import crispr_deephf.runner
    [as r]``, including the parenthesised multi-line form the setup cell uses.
    """
    out: set[str] = set()
    for cell in _code_cells(nb):
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "crispr_deephf"
            ):
                out |= {alias.asname or alias.name for alias in node.names}
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "crispr_deephf":
                        out.add(alias.asname or alias.name.split(".")[-1])
    return out


def test_package_module_aliases_are_imported():
    """Every ``mod.attr(...)`` call must have a real import behind it.

    The old version of this test derived both sides from the notebook's own source (it asked
    only whether the alias appeared somewhere in the text, which it had to, being the thing
    that was just read out of it), so it could not fail. This parses the import *statements*
    and compares the two sets.
    """
    nb = _nb()
    used = {mod for mod, _, _, _ in _module_calls(nb)}
    imported = _imported_package_aliases(nb)
    assert used, "no package calls found at all; the alias map or the walker is broken"
    assert "runner" in imported, "the import parser found no package imports"
    assert used <= imported, f"called but never imported: {sorted(used - imported)}"


def _module_bindings(name: str) -> dict[str, ast.AST]:
    """Names bound at module level in ``src/crispr_deephf/<name>.py``, as AST nodes.

    Parsed rather than imported, so this runs in CI, which installs no torch and could
    not import ``runner`` / ``architectures`` / ``tuning`` at all. Descends into
    ``try``/``if`` bodies (that is where ``VIENNA_AVAILABLE`` and ``OPTUNA_AVAILABLE``
    are set) but never into a function body.
    """
    path = SRC / f"{name}.py"
    assert path.is_file(), f"no such module: {path}"
    out: dict[str, ast.AST] = {}

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                out[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out[node.target.id] = node
            elif isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    out[(alias.asname or alias.name).split(".")[0]] = node
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)

    visit(ast.parse(path.read_text(encoding="utf-8")).body)
    return out


def _signature_from_ast(node: ast.AST) -> inspect.Signature | None:
    """Rebuild a function's signature from its ``def`` line, or None if it cannot be trusted.

    Returns None for anything decorated (a decorator can rewrite the signature) and for
    non-callables such as module constants, which the caller then skips.
    """
    if isinstance(node, ast.ClassDef):
        if node.decorator_list:
            return None  # e.g. @dataclass synthesises __init__
        init = next(
            (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        return None if init is None else _signature_from_ast_function(init, drop_first=True)
    if isinstance(node, ast.FunctionDef) and not node.decorator_list:
        return _signature_from_ast_function(node)
    return None


def _signature_from_ast_function(
    node: ast.FunctionDef, *, drop_first: bool = False
) -> inspect.Signature:
    kind = inspect.Parameter
    args = node.args
    names = [*args.posonlyargs, *args.args]
    kinds = [kind.POSITIONAL_ONLY] * len(args.posonlyargs) + [
        kind.POSITIONAL_OR_KEYWORD
    ] * len(args.args)
    if drop_first and names:  # 'self'
        names, kinds = names[1:], kinds[1:]
    # Defaults always align to the tail of the positional list, and 'self' never has one.
    n_required = len(names) - len(args.defaults)
    params = [
        kind(a.arg, k, default=kind.empty if i < n_required else None)
        for i, (a, k) in enumerate(zip(names, kinds, strict=True))
    ]
    if args.vararg:
        params.append(kind(args.vararg.arg, kind.VAR_POSITIONAL))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append(
            kind(arg.arg, kind.KEYWORD_ONLY, default=kind.empty if default is None else None)
        )
    if args.kwarg:
        params.append(kind(args.kwarg.arg, kind.VAR_KEYWORD))
    return inspect.Signature(params)


def test_every_package_call_matches_the_real_signature():
    """Catch a typo'd keyword here, not three hours into the user's Colab run.

    Deliberately torch-free: the package modules are *parsed*, not imported, so this test
    runs in CI (which installs the dev extra only) instead of silently skipping there.
    """
    bindings = {name: _module_bindings(name) for name in PACKAGE_MODULES}
    checked = 0
    for mod, attr, node, _ in _module_calls(_nb()):
        assert attr in bindings[mod], f"crispr_deephf.{mod} has no attribute {attr!r}"
        sig = _signature_from_ast(bindings[mod][attr])
        if sig is None:
            continue  # not a plain function/class we can rebuild
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            continue  # unpacked call: cannot be checked statically
        try:
            sig.bind(*[object()] * len(node.args), **{k.arg: object() for k in node.keywords})
        except TypeError as exc:
            pytest.fail(f"{mod}.{attr}{sig} called wrongly in the notebook: {exc}")
        checked += 1
    assert checked >= 25, f"only {checked} package calls checked; the alias map looks wrong"


def test_the_signature_check_rebuilds_signatures_correctly():
    """Guard the guard: the AST-rebuilt signature must match what Python itself reports."""
    pytest.importorskip("torch")  # runner / tuning cannot be imported without it
    for name in PACKAGE_MODULES:
        module = importlib.import_module(f"crispr_deephf.{name}")
        for attr, node in _module_bindings(name).items():
            rebuilt = _signature_from_ast(node)
            if rebuilt is None or not hasattr(module, attr):
                continue
            real = inspect.signature(getattr(module, attr))
            assert [(p.name, p.kind, p.default is p.empty) for p in rebuilt.parameters.values()] \
                == [(p.name, p.kind, p.default is p.empty) for p in real.parameters.values()], (
                    f"AST signature for {name}.{attr} disagrees with the real one: "
                    f"{rebuilt} vs {real}"
                )


def _cell1_detect_colab():
    """Run the setup cell's own detection code and return its ``detect_colab``.

    Only the part of cell 1 above the ``[A] GOOGLE COLAB`` banner is executed, so nothing
    tries to mount Drive. The function returned is the notebook's, not a copy.
    """
    head = _nb().cells[1].source.split("# [A] GOOGLE COLAB")[0]
    namespace: dict = {}
    exec(compile(head, "<setup-cell-head>", "exec"), namespace)  # noqa: S102
    assert isinstance(namespace["IN_COLAB"], bool)
    return namespace["detect_colab"]


@pytest.mark.parametrize(
    ("spec_result", "expected"),
    [
        ("raise", False),  # no `google` package at all: a clean conda env
        (None, False),  # a `google` namespace package without `colab` (protobuf, Cloud SDK)
        ("spec", True),  # real Colab
    ],
)
def test_colab_detection_handles_every_find_spec_outcome(monkeypatch, spec_result, expected):
    """The guard that keeps cell 1 from dying locally on ``No module named 'google'``.

    ``importlib.util.find_spec("google.colab")`` imports the parent ``google`` package
    before it can look inside it, and *raises* ModuleNotFoundError when nothing provides
    ``google`` -- the normal state of a clean conda env. Unguarded, the notebook's first
    cell crashed before printing a line; this was found by executing it, not by reading it.

    ``find_spec`` is monkeypatched rather than trusted, so the test proves the guard on
    every machine. Asserting only that the real call returns a bool would pass vacuously
    wherever any ``google.*`` distribution happens to be installed -- including on Colab
    itself, and possibly in CI.
    """
    detect = _cell1_detect_colab()

    def fake_find_spec(name, package=None):
        assert name == "google.colab"
        if spec_result == "raise":
            raise ModuleNotFoundError("No module named 'google'")
        return object() if spec_result == "spec" else None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert detect() is expected


def test_colab_detection_does_not_swallow_an_unrelated_error():
    """The guard catches a missing parent package, not every failure in the lookup."""
    head = _nb().cells[1].source.split("# [A] GOOGLE COLAB")[0]
    assert "except ModuleNotFoundError" in head
    assert "except Exception" not in head and "except:" not in head


# --------------------------------------------------------------------------- #
# The WT-only switch (ENZYMES -> ACTIVE_TARGETS)
# --------------------------------------------------------------------------- #
ENZYME_KEYS = {"WT": "WT_Efficiency", "eSp": "eSpCas9_Efficiency", "HF1": "SpCas9_HF1_Efficiency"}


def _cell_with(text: str) -> str:
    """The one code cell whose source contains `text` (fails if there are none or many)."""
    hits = [c.source for c in _code_cells(_nb()) if text in c.source]
    assert len(hits) == 1, f"expected exactly one code cell containing {text!r}, got {len(hits)}"
    return hits[0]


def _cell_function(cell_src: str, name: str) -> str:
    """Source of the top-level function `name` in a cell, so it can be exec'd on its own."""
    for node in ast.parse(cell_src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(cell_src, node)
    raise AssertionError(f"no top-level def {name} in the cell")


def _exec_cell_function(cell_src: str, name: str, namespace: dict):
    code = compile(_cell_function(cell_src, name), f"<notebook {name}>", "exec")
    exec(code, namespace)  # noqa: S102
    return namespace[name]


def _run_cell(cell_src: str, namespace: dict) -> dict:
    exec(compile(cell_src, "<notebook cell>", "exec"), namespace)  # noqa: S102
    return namespace


def _resolver():
    from crispr_deephf import config

    return _exec_cell_function(
        _cell_with("def resolve_active_targets"), "resolve_active_targets", {"config": config}
    )


def test_enzyme_switch_is_a_code_cell_setting_that_defaults_to_wt_only():
    assigned = _notebook_assignments(_nb())
    assert assigned["ENZYMES"] == ["WT"], "the shipped switch must be WT-SpCas9 only"
    # The switch lives in the settings cell, beside MODE and the RUN_* switches.
    settings = _cell_with('MODE = "quick"')
    assert "ENZYMES = " in settings
    assert "ACTIVE_TARGETS = resolve_active_targets(ENZYMES)" in settings
    # ...and resolves to the target column, via the real config mapping.
    assert _resolver()(assigned["ENZYMES"]) == ["WT_Efficiency"]


def test_enzyme_keys_the_switch_accepts_are_exactly_the_configs():
    from crispr_deephf import config

    assert {key: col for col, key in config.TARGET_KEYS.items()} == ENZYME_KEYS


@pytest.mark.parametrize(
    "setting, expected",
    [
        (["WT"], ["WT_Efficiency"]),
        (["HF1"], ["SpCas9_HF1_Efficiency"]),
        (["HF1", "WT"], ["WT_Efficiency", "SpCas9_HF1_Efficiency"]),  # canonical order
        (("eSp", "eSp"), ["eSpCas9_Efficiency"]),                      # duplicates collapse
        (["WT", "eSp", "HF1"], list(ENZYME_KEYS.values())),
    ],
)
def test_enzyme_switch_resolves_keys_to_target_columns(setting, expected):
    assert _resolver()(setting) == expected


@pytest.mark.parametrize(
    "bad",
    [["WTT"], ["wt"], ["WT", "Cas12"], [], "WT", None, ["WT_Efficiency"], [1], [["WT"]]],
)
def test_enzyme_switch_rejects_anything_that_is_not_a_valid_key_list(bad):
    """A typo must stop the run, and the error must list the valid keys."""
    with pytest.raises(ValueError) as err:
        _resolver()(bad)
    message = str(err.value)
    for key in ENZYME_KEYS:
        assert key in message, f"the error for {bad!r} must list the valid key {key!r}"


def test_no_cell_iterates_every_enzyme_except_the_eda_table():
    """Only cell 15 (the free EDA table) may walk all of config.TARGET_COLS / TARGET_KEYS.

    Read from the AST, so a comment or a printed message that mentions them cannot trip it.
    A bare lookup such as ``config.TARGET_KEYS[t]`` is fine; a loop or comprehension over the
    whole mapping is what the switch has to override.
    """
    iterating = []
    for cell in _code_cells(_nb()):
        tree = ast.parse(cell.source)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            whole = (
                isinstance(node, ast.Attribute)
                and node.attr in ("TARGET_COLS", "TARGET_KEYS")
                and isinstance(node.value, ast.Name)
                and node.value.id == "config"
            )
            if not whole:
                continue
            parent = parents[node]
            looked_up = isinstance(parent, ast.Subscript) and parent.value is node
            if looked_up:
                # config.TARGET_KEYS[...] is a lookup; but TARGET_COLS[0] picks "the first
                # enzyme", which is not the switch's first.
                if node.attr == "TARGET_COLS":
                    iterating.append(ast.unparse(parent))
                continue
            if "counts_vs_paper" in cell.source:
                continue   # cell 15: EDA over all three, labelled "Trained in this run"
            if "def resolve_active_targets" in cell.source:
                continue   # cell 5: the one place the keys are mapped onto the columns
            iterating.append(ast.unparse(parent)[:80])
    assert not iterating, f"these still ignore the ENZYMES switch: {iterating}"


def test_eda_cell_keeps_all_three_enzymes_and_says_which_are_trained():
    """Cell 15 stays three-enzyme (free, and the paper comparison needs it) but is labelled."""
    import io
    from contextlib import redirect_stdout

    import pandas as pd

    from crispr_deephf import config

    cell = _cell_with("counts_vs_paper = pd.DataFrame(rows)")
    df = pd.DataFrame({t: [0.5, None, 0.2] for t in config.TARGET_COLS})
    for active in (["WT_Efficiency"], list(config.TARGET_COLS)):
        ns = {"config": config, "pd": pd, "df": df, "ACTIVE_TARGETS": active,
              "ACTIVE_LABELS": [config.TARGET_LABELS[t] for t in active], "ENZYMES": ["WT"]}
        with redirect_stdout(io.StringIO()):
            _run_cell(cell, ns)
        table = ns["counts_vs_paper"]
        assert list(table["Enzyme"]) == [config.TARGET_LABELS[t] for t in config.TARGET_COLS]
        trained = dict(zip(table["Enzyme"], table["Trained in this run"], strict=True))
        for t in config.TARGET_COLS:
            assert trained[config.TARGET_LABELS[t]].startswith("yes" if t in active else "no")


def test_bio_prewarm_folds_only_the_active_targets_row_sets(synthetic_df):
    """Run the cell's own helper: with WT only, only the WT row set is built, never MULTI."""
    from crispr_deephf import config, data, runner

    df = data.add_shared_split(data.add_split_labels(synthetic_df))
    ns = {"config": config, "data": data, "runner": runner}
    helper = _exec_cell_function(_cell_with("def bio_prewarm_frames"), "bio_prewarm_frames", ns)

    wt = helper(df, ["WT_Efficiency"], with_multitask=False)
    assert list(wt) == ["WT"]
    assert len(wt["WT"]) == int(df["WT_Efficiency"].notna().sum())

    two = helper(df, ["WT_Efficiency", "SpCas9_HF1_Efficiency"], with_multitask=False)
    assert list(two) == ["WT", "HF1"]            # eSp is never folded

    everything = helper(df, list(config.TARGET_COLS), with_multitask=True)
    assert list(everything) == ["WT", "eSp", "HF1", runner.MULTI_KEY]
    keep = df[config.SHARED_SPLIT_COL].notna() & df[config.TARGET_COLS].notna().any(axis=1)
    assert len(everything[runner.MULTI_KEY]) == int(keep.sum())


def test_bio_prewarm_cell_builds_its_frames_from_the_switch_not_from_all_targets():
    """The cell must hand the helper ACTIVE_TARGETS and gate MULTI on the multi-task skip."""
    cell = _cell_with("def bio_prewarm_frames")
    calls = [
        n for n in ast.walk(ast.parse(cell))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "bio_prewarm_frames"
    ]
    assert len(calls) == 1
    args = [ast.unparse(a) for a in calls[0].args]
    assert args[1] == "ACTIVE_TARGETS"
    assert "MULTITASK_SKIP_REASON" in args[2] and "RUN_EXTRAS" in args[2]


def _multitask_namespace(active_targets, *, run_extras=True, vienna=True):
    from types import SimpleNamespace

    from crispr_deephf import config

    calls = []
    skip = _exec_cell_function(
        _cell_with("def multitask_skip_reason"), "multitask_skip_reason",
        {"ACTIVE_KEYS": [config.TARGET_KEYS[t] for t in active_targets],
         "MULTITASK_N_TARGETS": 3},
    )
    ns = {
        "config": config,
        "RUN_EXTRAS": run_extras,
        "ENZYMES": [config.TARGET_KEYS[t] for t in active_targets],
        "ACTIVE_TARGETS": list(active_targets),
        "MULTITASK_MODEL": "MultiTask-BiLSTM+bio",
        "MULTITASK_SKIP_REASON": skip(active_targets),
        "bio_features": SimpleNamespace(VIENNA_AVAILABLE=vienna),
        "runner": SimpleNamespace(run_deep_model=lambda *a, **k: calls.append(k) or [k]),
        "df": None, "paths": None, "MODE": "quick", "SEEDS": [0], "DEVICE": "cpu",
        "FORCE_RERUN": False,
    }
    return ns, calls


def test_multitask_cell_is_skipped_with_a_reason_when_one_enzyme_is_active(capsys):
    ns, calls = _multitask_namespace(["WT_Efficiency"])
    _run_cell(_cell_with("multitask_results = []"), ns)
    out = capsys.readouterr().out
    assert calls == [], "the multi-task model must not be handed to run_deep_model"
    assert ns["multitask_results"] == []
    assert "SKIPPED" in out and "only 1 is selected" in out, "it must say why"


def test_multitask_cell_is_skipped_for_a_partial_subset_too(capsys):
    ns, calls = _multitask_namespace(["WT_Efficiency", "eSpCas9_Efficiency"])
    _run_cell(_cell_with("multitask_results = []"), ns)
    out = capsys.readouterr().out
    assert calls == [] and "SKIPPED" in out and "3 outputs" in out


def test_multitask_cell_still_runs_with_all_three_enzymes_on_the_shared_split():
    from crispr_deephf import config

    ns, calls = _multitask_namespace(list(config.TARGET_COLS))
    _run_cell(_cell_with("multitask_results = []"), ns)
    assert len(calls) == 1
    assert calls[0]["target_cols"] == list(config.TARGET_COLS)
    assert calls[0]["split_col"] == config.SHARED_SPLIT_COL


def test_multitask_cell_does_not_announce_a_skip_when_extras_are_off(capsys):
    ns, calls = _multitask_namespace(["WT_Efficiency"], run_extras=False)
    _run_cell(_cell_with("multitask_results = []"), ns)
    out = capsys.readouterr().out
    assert calls == [] and "RUN_EXTRAS = False" in out and "SKIPPED" not in out


def test_tier1_and_figures_are_handed_the_active_targets():
    """The package calls that otherwise default to all three enzymes get the switch."""
    seen = {}
    for mod, attr, node, _ in _module_calls(_nb()):
        if (mod, attr) in {
            ("runner", "run_tier1"), ("plots", "plot_vs_paper"),
            ("plots", "plot_tier_progression"), ("plots", "plot_pred_vs_actual"),
            ("plots", "plot_residuals"),
        }:
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
            seen[attr] = kw.get("targets")
    assert seen == dict.fromkeys(
        ("run_tier1", "plot_vs_paper", "plot_tier_progression", "plot_pred_vs_actual",
         "plot_residuals"),
        "ACTIVE_TARGETS",
    )


# --------------------------------------------------------------------------- #
# Section 6: the measured Colab cost
# --------------------------------------------------------------------------- #
def _bio_cost_cell() -> str:
    hits = [c.source for c in _nb().cells
            if c.cell_type == "markdown" and c.source.startswith("## 6. Bio features")]
    assert len(hits) == 1
    return hits[0]


def test_cost_note_states_both_measurements_with_attribution():
    import re

    text = _bio_cost_cell()
    assert re.search(r"Desktop.{0,40}38\.7 ms", text, re.DOTALL)
    assert re.search(r"Colab A100.{0,60}70 ms", text, re.DOTALL)
    assert "6,000 guides" in text and "7 minutes" in text
    assert "1.81x" in text
    assert "we have not measured one" not in text, "the stale 'not measured' claim must go"


def test_cost_note_says_the_gpu_does_not_help():
    import re

    text = _bio_cost_cell()
    assert "A100 does not help" in text
    assert "single-threaded" in text and "CPU-bound" in text
    assert re.search(r"GPU\s+is\s+not\s+involved.{0,80}idle", text, re.DOTALL)
    assert "faster GPU tier buys nothing" in text


def test_cost_note_arithmetic_is_reproducible():
    """Every `<ms> ms x <guides> (<label>) = <n> min` line must actually multiply out."""
    import re

    text = _bio_cost_cell()
    rows = re.findall(
        r"^\s*([\d.]+) ms x\s+([\d,]+) \(([^)]*)\)\s*=\s*([\d,]+) min", text, re.MULTILINE
    )
    assert len(rows) >= 11, "per-file lines for both machines, plus totals"
    for ms, guides, label, minutes in rows:
        got = float(ms) * int(guides.replace(",", "")) / 60_000
        assert round(got) == int(minutes.replace(",", "")), (label, ms, guides, minutes, got)
    colab = {label: int(g.replace(",", "")) for ms, g, label, _ in rows if ms == "70"}
    assert colab["WT only"] == colab["bio_WT"] == 55_604
    assert colab["all four"] == 230_961 == sum(
        colab[k] for k in ("bio_WT", "bio_eSp", "bio_HF1", "bio_MULTI")
    )
    assert 70 / 38.7 == pytest.approx(1.81, abs=0.005)   # the stated slow-down
    assert 7 * 60 / 6000 * 1000 == pytest.approx(70)       # the stated measurement
