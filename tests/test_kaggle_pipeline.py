import json
import ast
from pathlib import Path

from scripts.run_kaggle_pipeline import (
    PipelineRunner,
    build_jobs,
    find_artifact_root,
    materialize_upload,
    parse_pipeline_status,
    parse_status,
)


def _config(tmp_path):
    return {
        "kaggle": {
            "owner": "test-owner",
            "raw_dataset": "test-owner/raw-eeg",
            "work_dir": str(tmp_path / "work"),
            "state_path": str(tmp_path / "state.json"),
            "poll_interval_seconds": 1,
        },
        "stages": ["00", "01", "02", "03", "04", "05", "06", "07", "08"],
        "matrix": {
            "datasets": ["eyesclosed", "photomark"],
            "experiments": ["baseline", "no_asr"],
            "feature_sets": ["full"],
        },
    }


def test_status_parser_handles_kaggle_output():
    assert parse_status("Status: Running") == "running"
    assert parse_status("Status: Complete") == "complete"
    assert parse_status("Status: Errored") == "failed"


def test_build_jobs_is_stage_major():
    jobs = build_jobs(_config(Path("/tmp")))
    assert jobs[0].job_id == "00:all:baseline:full"
    assert jobs[1].job_id == "01:eyesclosed:baseline:full"
    assert jobs[2].job_id == "01:eyesclosed:no_asr:full"
    assert jobs[-1].job_id == "08:photomark:aggregate:full"


def test_render_kernel_embeds_run_config_and_metadata(tmp_path):
    runner = PipelineRunner(_config(tmp_path), Path("configs/kaggle_pipeline.yaml"))
    job = build_jobs(_config(tmp_path))[1]
    kernel_dir, kernel_id = runner.render_kernel(job, 1, "test-owner/preprocessed")

    notebook = json.loads(next(kernel_dir.glob("*.ipynb")).read_text())
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    assert "'dataset': 'eyesclosed'" in code
    assert "'pipeline_input': 'test-owner/preprocessed'" in code
    assert metadata["id"] == kernel_id
    assert metadata["dataset_sources"] == [
        "test-owner/raw-eeg",
        "test-owner/preprocessed",
    ]


def test_materialize_upload_finds_pipeline_data_and_status(tmp_path):
    download = tmp_path / "download" / "pipeline_output" / "data"
    (download / "features" / "eyesclosed" / "baseline").mkdir(parents=True)
    (download / "pipeline_status.json").write_text(
        json.dumps({"status": "partial"}), encoding="utf-8"
    )
    (download / "features" / "eyesclosed" / "baseline" / "marker.txt").write_text(
        "ok", encoding="utf-8"
    )
    assert find_artifact_root(tmp_path / "download") == download
    assert parse_pipeline_status(tmp_path / "download")["status"] == "partial"

    job = build_jobs(_config(tmp_path))[3]
    upload = materialize_upload(
        tmp_path / "download",
        tmp_path / "upload",
        "test-owner/output",
        job,
    )
    assert (upload / "data" / "features" / "eyesclosed" / "baseline" / "marker.txt").is_file()
    metadata = json.loads((upload / "dataset-metadata.json").read_text())
    assert metadata["id"] == "test-owner/output"


def test_runner_publishes_only_after_kernel_completion(tmp_path):
    config = _config(tmp_path)
    config["stages"] = ["01"]
    config["matrix"]["datasets"] = ["eyesclosed"]
    config["matrix"]["experiments"] = ["baseline"]

    class FakeCli:
        def __init__(self):
            self.status_calls = 0
            self.created = 0

        def status(self, _kernel_id):
            self.status_calls += 1
            if self.status_calls == 1:
                return "Status: Not Found"
            return "Status: Running" if self.status_calls == 2 else "Status: Complete"

        def push(self, _kernel_dir, _timeout):
            return "pushed"

        def logs(self, _kernel_id):
            return "logs"

        def output(self, _kernel_id, destination):
            data = destination / "pipeline_output" / "data" / "preprocessed" / "eyesclosed" / "baseline"
            data.mkdir(parents=True)
            (data / "metadata.json").write_text("{}", encoding="utf-8")
            (destination / "pipeline_output" / "data" / "pipeline_status.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            return "downloaded"

        def dataset_create(self, _upload_dir):
            self.created += 1
            return "created"

        def dataset_version(self, _upload_dir, _message):
            raise AssertionError("first publication should create the Dataset")

        def dataset_status(self, _dataset_id):
            return "Status: Complete"

    runner = PipelineRunner(config, Path("configs/kaggle_pipeline.yaml"))
    runner.cli = FakeCli()
    runner.run()

    state = json.loads((tmp_path / "state.json").read_text())
    record = state["jobs"]["01:eyesclosed:baseline:full"]
    assert record["status"] == "complete"
    assert record["dataset_id"] == "test-owner/ad-eeg-pipeline-eyesclosed-full-preprocessed"
    assert runner.cli.created == 1
