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
