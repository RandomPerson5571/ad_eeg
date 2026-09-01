"""Ensure generated Kaggle notebooks request the correct upstream input type."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _code(name: str) -> str:
    notebook = json.loads((ROOT / "notebooks" / "kaggle" / name).read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _pipeline_locator(name: str, mount: Path):
    code = _code(name)
    tree = ast.parse(code)
    locator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pipeline_data_source"
    )
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[locator], type_ignores=[]), "<locator>", "exec"), namespace)
    namespace["_input_mount_candidates"] = lambda _: [mount]
    return namespace["_pipeline_data_source"]("ignored")


def test_epoching_consumes_pipeline_output_not_raw_eeg():
    code = _code("02_epoching.ipynb")
    assert "RAW_EEG_INPUT = None" in code
    assert 'PIPELINE_INPUT = "REPLACE_WITH_PRIOR_PIPELINE_OUTPUT_SLUG"' in code
    assert "REQUIRES_PIPELINE_INPUT = True" in code
    assert "validate_epoch_exports(paths)" in code


def test_preprocessing_publishes_a_validated_stage_contract():
    code = _code("01_preprocessing.ipynb")
    assert 'RAW_EEG_INPUT = "REPLACE_WITH_RAW_EEG_DATASET_SLUG"' in code
    assert "PIPELINE_INPUT = None" in code
    assert "validate_preprocessed_artifacts" in code


def test_epoching_accepts_current_preprocessing_artifact_tree(tmp_path):
    root = tmp_path / "pipeline_output" / "data"
    epoch = root / "preprocessed" / "eyesclosed" / "baseline" / "sub-001_epo.fif"
    epoch.parent.mkdir(parents=True)
    epoch.touch()

    assert _pipeline_locator("02_epoching.ipynb", tmp_path) == (root, "data_tree")


def test_epoching_accepts_direct_preprocessing_artifact_tree(tmp_path):
    root = tmp_path / "pipeline_output"
    epoch = root / "eyesclosed" / "baseline" / "sub-001_epo.fif"
    epoch.parent.mkdir(parents=True)
    epoch.touch()

    assert _pipeline_locator("02_epoching.ipynb", tmp_path) == (
        root,
        "preprocessed_tree",
    )


def test_epoching_accepts_kaggle_wrapper_around_preprocessing_tree(tmp_path):
    root = tmp_path / "saved-output" / "pipeline_output"
    epoch = root / "dataset2" / "baseline" / "sub-001_epo.fif"
    epoch.parent.mkdir(parents=True)
    epoch.touch()

    assert _pipeline_locator("02_epoching.ipynb", tmp_path) == (
        root,
        "preprocessed_tree",
    )
