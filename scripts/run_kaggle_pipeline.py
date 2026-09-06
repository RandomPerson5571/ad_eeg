#!/usr/bin/env python3
"""Run the Kaggle notebook pipeline as a resumable serial workflow.

The runner deliberately keeps scheduling outside Kaggle. It pushes one kernel,
waits for that kernel, publishes its output as a Dataset version, and only then
starts the next job. A JSON state file makes the workflow safe to resume from a
cron/launchd/GitHub Actions invocation after a local interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "00": "00_dataset_audit.ipynb",
    "01": "01_preprocessing.ipynb",
    "02": "02_epoching.ipynb",
    "03": "03_feature_extraction.ipynb",
    "04": "04_feature_selection.ipynb",
    "05": "05_classical_ml.ipynb",
    "06": "06_deep_learning.ipynb",
    "07": "07_ablation.ipynb",
    "08": "08_final_benchmark.ipynb",
}
STAGE_NAMES = {
    "00": "00_dataset_audit",
    "01": "01_preprocessing",
    "02": "02_epoching",
    "03": "03_feature_extraction",
    "04": "04_feature_selection",
    "05": "05_classical_ml",
    "06": "06_deep_learning",
    "07": "07_ablation",
    "08": "08_final_benchmark",
}
RAW_STAGES = {"00", "01"}
OUTPUT_FAMILY = {
    "00": "audit",
    "01": "preprocessed",
    "02": "epochs",
    "03": "features",
    "04": "selected",
    "05": "results",
    "06": "results",
    "07": "results",
    "08": "results",
}
INPUT_FAMILY = {
    "01": "preprocessed",
    "02": "preprocessed",
    "03": "epochs",
    "04": "features",
    "05": "selected",
    "06": "results",
    "07": "results",
    "08": "results",
}
STAGE_ALIASES = {
    name: key for key, name in STAGE_NAMES.items()
} | {str(key): str(key) for key in STAGE_NAMES}


class KaggleCommandError(RuntimeError):
    """A Kaggle CLI command failed."""

    def __init__(self, command: list[str], output: str, returncode: int):
        self.command = command
        self.output = output
        self.returncode = returncode
        rendered = " ".join(command)
        super().__init__(f"Kaggle command failed ({returncode}): {rendered}\n{output}")


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    dataset: str
    experiment: str
    feature_set: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, max_length: int = 48) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return (value or "run")[:max_length].strip("-")


def expand_env(value: Any) -> Any:
    """Expand ${VAR} in YAML strings without exposing secrets in logs."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda match: os.environ.get(match.group(1), match.group(0)),
            value,
        )
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = expand_env(yaml.safe_load(handle) or {})
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML object: {path}")
    return config


def canonical_stage(value: str) -> str:
    value = str(value).strip()
    if value in STAGE_ALIASES:
        return STAGE_ALIASES[value]
    raise ValueError(
        f"Unknown Kaggle stage {value!r}; choose one of: "
        + ", ".join(STAGE_NAMES.values())
    )


def matrix_values(config: dict[str, Any], key: str) -> list[str]:
    values = config.get("matrix", {}).get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"matrix.{key} must be a non-empty list")
    return [str(value) for value in values]


def build_jobs(config: dict[str, Any]) -> list[Job]:
    """Create a deterministic stage-major plan.

    Notebook 00 audits the raw source once. Stages 01–06 run for every matrix
    combination. Stages 07–08 aggregate all experiments for each dataset.
    """
    datasets = matrix_values(config, "datasets")
    experiments = matrix_values(config, "experiments")
    feature_sets = matrix_values(config, "feature_sets")
    stages = [canonical_stage(value) for value in config.get("stages", NOTEBOOKS)]
    jobs: list[Job] = []
    for stage in stages:
        if stage == "00":
            jobs.append(
                Job(
                    job_id=f"{stage}:all:{experiments[0]}:{feature_sets[0]}",
                    stage=stage,
                    dataset="all",
                    experiment=experiments[0],
                    feature_set=feature_sets[0],
                )
            )
        elif stage in {"07", "08"}:
            for dataset in datasets:
                for feature_set in feature_sets:
                    jobs.append(
                        Job(
                            job_id=f"{stage}:{dataset}:aggregate:{feature_set}",
                            stage=stage,
                            dataset=dataset,
                            experiment="aggregate",
                            feature_set=feature_set,
                        )
                    )
        else:
            for dataset in datasets:
                for experiment in experiments:
                    for feature_set in feature_sets:
                        jobs.append(
                            Job(
                                job_id=f"{stage}:{dataset}:{experiment}:{feature_set}",
                                stage=stage,
                                dataset=dataset,
                                experiment=experiment,
                                feature_set=feature_set,
                            )
                        )
    return jobs


def parse_status(output: str) -> str:
    """Normalize Kaggle's human-readable status output."""
    status_lines = []
    for line in output.splitlines():
        lowered = line.lower()
        if "status" in lowered or "state" in lowered:
            status_lines.append(lowered.split(":", 1)[-1].strip())
    haystack = " ".join(status_lines or [output.lower()])
    if any(token in haystack for token in ("failed", "errored", "error", "cancelled", "canceled")):
        return "failed"
    if any(
        token in haystack
        for token in (
            "complete",
            "completed",
            "success",
            "succeeded",
            "finished",
            "ready",
            "published",
            "available",
        )
    ):
        return "complete"
    if any(token in haystack for token in ("running", "queued", "pending", "starting", "waiting")):
        return "running"
    return "unknown"


def parse_pipeline_status(download_dir: Path) -> dict[str, Any]:
    candidates = [
        download_dir / "pipeline_output" / "data" / "pipeline_status.json",
        download_dir / "data" / "pipeline_status.json",
        download_dir / "pipeline_status.json",
    ]
    candidates.extend(download_dir.rglob("pipeline_status.json"))
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Pipeline status must be a JSON object: {candidate}")
            status = str(payload.get("status", "complete")).lower()
            if status not in {"partial", "complete"}:
                raise ValueError(f"Unsupported pipeline status {status!r}: {candidate}")
            payload["status"] = status
            return payload
    # Older generated notebooks did not write the marker. A successful Kaggle
    # kernel is still safe to publish; the marker is required only to detect
    # notebook 03's resumable PARTIAL result.
    return {"status": "complete"}


def contains_artifact_tree(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).is_dir()
        for name in ("audit", "preprocessed", "features", "models", "results")
    )


def find_artifact_root(download_dir: Path) -> Path:
    candidates = [
        download_dir / "pipeline_output" / "data",
        download_dir / "data",
        download_dir / "pipeline_output",
        download_dir,
    ]
    candidates.extend(download_dir.rglob("data"))
    for candidate in candidates:
        if contains_artifact_tree(candidate):
            return candidate
    raise FileNotFoundError(
        f"Kaggle output {download_dir} does not contain a recognized data tree"
    )


def materialize_upload(
    download_dir: Path,
    upload_dir: Path,
    dataset_id: str,
    job: Job,
    dataset_license: str = "other",
) -> Path:
    source = find_artifact_root(download_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    data_target = upload_dir / "data"
    shutil.copytree(source, data_target, dirs_exist_ok=True)
    for leaked_repo in upload_dir.rglob(".git"):
        if leaked_repo.is_dir():
            shutil.rmtree(leaked_repo)
    metadata = {
        "title": f"AD EEG {job.dataset} {OUTPUT_FAMILY[job.stage]}",
        "subtitle": "Resumable staged pipeline artifact",
        "description": (
            f"Generated by {STAGE_NAMES[job.stage]} for dataset={job.dataset}, "
            f"experiment={job.experiment}, feature_set={job.feature_set}."
        ),
        "id": dataset_id,
        "licenses": [{"name": dataset_license}],
    }
    (upload_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return upload_dir


class KaggleCli:
    def __init__(self, executable: str = "kaggle"):
        self.executable = executable

    def run(self, args: Iterable[str], *, cwd: Path | None = None) -> str:
        command = [self.executable, *[str(arg) for arg in args]]
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = result.stdout or ""
        if result.returncode:
            raise KaggleCommandError(command, output, result.returncode)
        return output

    def push(self, kernel_dir: Path, timeout_seconds: int) -> str:
        return self.run(
            [
                "kernels",
                "push",
                "-p",
                kernel_dir,
                "-t",
                timeout_seconds,
            ]
        )

    def status(self, kernel_id: str) -> str:
        return self.run(["kernels", "status", kernel_id])

    def logs(self, kernel_id: str) -> str:
        return self.run(["kernels", "logs", kernel_id])

    def output(self, kernel_id: str, destination: Path) -> str:
        destination.mkdir(parents=True, exist_ok=True)
        return self.run(
            ["kernels", "output", kernel_id, "-p", destination, "-o"]
        )

    def dataset_status(self, dataset_id: str) -> str:
        return self.run(["datasets", "status", dataset_id, "--format", "json"])

    def dataset_create(self, upload_dir: Path) -> str:
        return self.run(
            [
                "datasets",
                "create",
                "-p",
                upload_dir,
                "-q",
                "-t",
                "-r",
                "tar",
            ]
        )

    def dataset_version(self, upload_dir: Path, message: str) -> str:
        return self.run(
            [
                "datasets",
                "version",
                "-p",
                upload_dir,
                "-m",
                message,
                "-q",
                "-t",
                "-r",
                "tar",
            ]
        )


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"version": 1, "jobs": {}, "handles": {}}
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = utc_now()
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def job(self, job_id: str) -> dict[str, Any]:
        return self.data.setdefault("jobs", {}).setdefault(job_id, {})


class PipelineRunner:
    def __init__(self, config: dict[str, Any], config_path: Path, *, dry_run: bool = False):
        self.config = config
        self.config_path = config_path
        self.dry_run = dry_run
        kaggle = config.get("kaggle", {})
        self.owner = str(kaggle.get("owner", "")).strip()
        if self.owner.startswith("REPLACE_WITH_") or self.owner.startswith("${"):
            self.owner = ""
        self.raw_dataset = str(kaggle.get("raw_dataset", "")).strip()
        self.repo_url = str(kaggle.get("repo_url", "https://github.com/RandomPerson5571/ad_eeg.git"))
        self.repo_branch = str(kaggle.get("repo_branch", "main"))
        self.kernel_prefix = str(kaggle.get("kernel_prefix", "ad-eeg"))
        self.dataset_prefix = str(kaggle.get("output_dataset_prefix", "ad-eeg-pipeline"))
        self.dataset_license = str(kaggle.get("dataset_license", "other"))
        self.work_dir = ROOT / str(kaggle.get("work_dir", ".kaggle_pipeline"))
        self.state = StateStore(ROOT / str(kaggle.get("state_path", ".kaggle_pipeline/state.json")))
        self.poll_interval = min(int(kaggle.get("poll_interval_seconds", 60)), 60)
        self.timeout_seconds = int(kaggle.get("timeout_seconds", 43200))
        self.max_partial_runs = int(kaggle.get("max_partial_runs", 32))
        self.cli = KaggleCli(str(kaggle.get("cli", "kaggle")))

    def validate(self) -> None:
        if not self.owner and not self.dry_run:
            raise ValueError(
                "Set kaggle.owner in configs/kaggle_pipeline.yaml or KAGGLE_USERNAME."
            )
        if (
            not self.raw_dataset
            or self.raw_dataset.startswith("REPLACE_WITH_")
            or "${" in self.raw_dataset
        ) and not self.dry_run:
            raise ValueError("Set kaggle.raw_dataset to the owner/raw-dataset handle.")
        if not (ROOT / "notebooks" / "kaggle").is_dir():
            raise FileNotFoundError("notebooks/kaggle directory is missing")
        if self.poll_interval < 1:
            raise ValueError("kaggle.poll_interval_seconds must be positive")

    def stage_options(self, stage: str) -> dict[str, Any]:
        options = self.config.get("stage_options", {})
        return dict(options.get(stage, options.get(STAGE_NAMES[stage], {})) or {})

    def effective_experiment(self, job: Job) -> str:
        if job.experiment != "aggregate":
            return job.experiment
        return matrix_values(self.config, "experiments")[0]

    def dataset_id(self, job: Job) -> str:
        family = OUTPUT_FAMILY[job.stage]
        slug = slugify(f"{self.dataset_prefix}-{job.dataset}-{job.feature_set}-{family}")
        return f"{self.owner}/{slug}"

    def kernel_id(self, job: Job) -> str:
        slug = slugify(
            f"{self.kernel_prefix}-{job.dataset}-{job.experiment}-{job.feature_set}-{job.stage}"
        )
        return f"{self.owner}/{slug}"

    def handle_key(self, job: Job, family: str) -> str:
        return f"{job.dataset}:{job.feature_set}:{family}"

    def input_dataset(self, job: Job) -> str | None:
        family = INPUT_FAMILY.get(job.stage)
        if not family:
            return None
        handle = self.state.data.get("handles", {}).get(self.handle_key(job, family))
        if handle:
            return str(handle)
        # Allow a user to start from a manually published artifact.
        configured = self.config.get("inputs", {}).get(family)
        if isinstance(configured, dict):
            configured = configured.get(job.dataset) or configured.get("default")
        return str(configured) if configured else None

    def render_kernel(self, job: Job, attempt: int, input_dataset: str | None) -> tuple[Path, str]:
        if not self.owner:
            raise ValueError("kaggle.owner is required to render a kernel")
        options = self.stage_options(job.stage)
        effective_experiment = self.effective_experiment(job)
        run_config = {
            "dataset": job.dataset,
            "experiment": effective_experiment,
            "feature_set": job.feature_set,
            "raw_eeg_input": self.raw_dataset if job.stage in RAW_STAGES else None,
            "pipeline_input": input_dataset,
            "mode": options.get("mode", "full"),
            "force": bool(options.get("force", False)),
            "workers": int(options.get("workers", 2)),
            "test_subjects": int(options.get("test_subjects", 5)),
            "inspect_subject": int(options.get("inspect_subject", 1)),
            "feature_subject_limit": options.get("feature_subject_limit", 24),
            "keep_intermediate_checkpoints": bool(
                options.get("keep_intermediate_checkpoints", False)
            ),
            "ablations": options.get("ablations", matrix_values(self.config, "experiments")),
            "repo_url": self.repo_url,
            "repo_branch": self.repo_branch,
        }
        attempt_dir = self.work_dir / "jobs" / slugify(job.job_id, 80) / f"attempt-{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        notebook_name = NOTEBOOKS[job.stage]
        source_path = ROOT / "notebooks" / "kaggle" / notebook_name
        notebook = json.loads(source_path.read_text(encoding="utf-8"))
        embedded = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if "RUN_CONFIG = {}" in source:
                source = source.replace(
                    "RUN_CONFIG = {}",
                    # The notebook cell is Python, so use repr rather than
                    # JSON (JSON's true/false/null are not Python literals).
                    f"RUN_CONFIG = {run_config!r}",
                    1,
                )
                cell["source"] = [line + "\n" for line in source.splitlines()]
                embedded = True
                break
        if not embedded:
            raise ValueError(
                f"{source_path} is not generated with the run-config hook; "
                "run scripts/generate_kaggle_notebooks.py first."
            )
        rendered_notebook = attempt_dir / notebook_name
        rendered_notebook.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        (attempt_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
        sources = []
        for source in (run_config["raw_eeg_input"], input_dataset):
            if source and source not in sources:
                sources.append(source)
        kernel_id = self.kernel_id(job)
        machine_shape = options.get("machine_shape", "")
        metadata = {
            "id": kernel_id,
            "title": slugify(f"{self.kernel_prefix}-{job.job_id}", 50),
            "code_file": notebook_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true" if machine_shape else "false",
            "enable_internet": "true",
            "machine_shape": machine_shape,
            "dataset_sources": sources,
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (attempt_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return attempt_dir, kernel_id

    def wait_for_kernel(
        self,
        kernel_id: str,
        log_path: Path,
        previous_output: str = "",
    ) -> str:
        deadline = time.monotonic() + self.timeout_seconds
        pushed_at = time.monotonic()
        last_output = ""
        while time.monotonic() <= deadline:
            try:
                last_output = self.cli.status(kernel_id)
            except KaggleCommandError as error:
                last_output = error.output
                if time.monotonic() + self.poll_interval > deadline:
                    raise
            log_path.write_text(last_output, encoding="utf-8")
            status = parse_status(last_output)
            if status in {"complete", "failed"}:
                # A push can briefly leave the previous successful run visible.
                # Give the new version time to enter the queue before accepting
                # an unchanged COMPLETE response as the current run.
                if (
                    status == "complete"
                    and previous_output
                    and last_output.strip() == previous_output.strip()
                    and time.monotonic() - pushed_at < 10
                ):
                    time.sleep(min(self.poll_interval, 5))
                    continue
                if status == "failed":
                    try:
                        last_output += "\n\n" + self.cli.logs(kernel_id)
                    except KaggleCommandError as error:
                        last_output += f"\n\nUnable to fetch logs: {error.output}"
                    log_path.write_text(last_output, encoding="utf-8")
                return status
            time.sleep(self.poll_interval)
        raise TimeoutError(f"Timed out waiting for {kernel_id}; see {log_path}")

    def publish(self, job: Job, dataset_id: str, download_dir: Path, record: dict[str, Any]) -> str:
        upload_dir = download_dir.parent / "dataset-upload"
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        materialize_upload(
            download_dir,
            upload_dir,
            dataset_id,
            job,
            dataset_license=self.dataset_license,
        )
        created = bool(record.get("dataset_created"))
        needs_version = False
        message = (
            f"{STAGE_NAMES[job.stage]}: dataset={job.dataset}, "
            f"experiment={self.effective_experiment(job)}, feature_set={job.feature_set}"
        )
        if not created:
            try:
                output = self.cli.dataset_create(upload_dir)
                record["dataset_create_output"] = output[-4000:]
                created = True
            except KaggleCommandError as error:
                # A previous process can have created the Dataset after its
                # state write was interrupted. Version it instead of failing.
                if not any(token in error.output.lower() for token in ("already exists", "duplicate", "conflict")):
                    raise
                created = True
                needs_version = True
                record["dataset_create_recovered"] = error.output[-4000:]
        if created and (record.get("dataset_versioned") or needs_version):
            output = self.cli.dataset_version(upload_dir, message)
            record["dataset_version_output"] = output[-4000:]
        elif created and record.get("dataset_created"):
            output = self.cli.dataset_version(upload_dir, message)
            record["dataset_version_output"] = output[-4000:]
        record["dataset_created"] = True
        record["dataset_versioned"] = True
        record["dataset_id"] = dataset_id
        record["upload_dir"] = str(upload_dir)
        record["dataset_status"] = self.wait_for_dataset(dataset_id)[-4000:]
        return dataset_id

    def wait_for_dataset(self, dataset_id: str) -> str:
        """Wait until Kaggle exposes the newly created Dataset version."""
        deadline = time.monotonic() + min(self.timeout_seconds, 900)
        last_output = ""
        while time.monotonic() <= deadline:
            last_output = self.cli.dataset_status(dataset_id)
            status = parse_status(last_output)
            if status == "complete":
                return last_output
            if status == "failed":
                raise RuntimeError(f"Kaggle Dataset publication failed: {dataset_id}\n{last_output}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"Timed out waiting for Dataset {dataset_id}: {last_output}")

    def run_job(self, job: Job) -> None:
        record = self.state.job(job.job_id)
        if record.get("status") == "complete":
            output_dataset = record.get("dataset_id")
            if output_dataset:
                self.state.data.setdefault("handles", {})[
                    self.handle_key(job, OUTPUT_FAMILY[job.stage])
                ] = output_dataset
            self.state.save()
            print(f"SKIP {job.job_id} (already complete)")
            return
        if record.get("status") == "failed" and not self.config.get("retry_failed", False):
            raise RuntimeError(
                f"{job.job_id} previously failed. Fix the issue and set retry_failed: true."
            )
        partial_runs = int(record.get("partial_runs", 0))
        while True:
            attempt = int(record.get("attempts", 0)) + 1
            input_dataset = self.input_dataset(job)
            output_dataset = self.dataset_id(job)
            kernel_dir, kernel_id = self.render_kernel(job, attempt, input_dataset)
            record.update(
                {
                    "status": "running",
                    "attempts": attempt,
                    "kernel_id": kernel_id,
                    "started_at": utc_now(),
                    "input_dataset": input_dataset,
                }
            )
            self.state.save()
            print(
                f"RUN {job.job_id} attempt={attempt} input={input_dataset or '<raw only>'}"
            )
            timeout = int(self.stage_options(job.stage).get("timeout_seconds", self.timeout_seconds))
            try:
                try:
                    previous_status = self.cli.status(kernel_id)
                except KaggleCommandError:
                    previous_status = ""
                push_output = self.cli.push(kernel_dir, timeout)
                record["push_output"] = push_output[-4000:]
                self.state.save()
                log_path = kernel_dir / "kernel-status.log"
                status = self.wait_for_kernel(kernel_id, log_path, previous_status)
                if status != "complete":
                    record.update({"status": "failed", "finished_at": utc_now()})
                    self.state.save()
                    raise RuntimeError(f"Kaggle kernel failed: {job.job_id}; see {log_path}")
                download_dir = kernel_dir / "kernel-output"
                output = self.cli.output(kernel_id, download_dir)
                record["output_download"] = output[-4000:]
                marker = parse_pipeline_status(download_dir)
                self.publish(job, output_dataset, download_dir, record)
            except Exception as error:
                record.update(
                    {
                        "status": "failed",
                        "finished_at": utc_now(),
                        "error": str(error)[-8000:],
                    }
                )
                self.state.save()
                raise

            handle_key = self.handle_key(job, OUTPUT_FAMILY[job.stage])
            self.state.data.setdefault("handles", {})[handle_key] = output_dataset
            record["dataset_id"] = output_dataset
            if marker["status"] == "partial":
                partial_runs += 1
                record.update(
                    {
                        "status": "partial",
                        "partial_runs": partial_runs,
                        "finished_at": utc_now(),
                    }
                )
                self.state.save()
                print(
                    f"PARTIAL {job.job_id}: published {output_dataset}; "
                    "rerunning the same notebook with the latest version"
                )
                if partial_runs >= self.max_partial_runs:
                    raise RuntimeError(
                        f"{job.job_id} exceeded max_partial_runs={self.max_partial_runs}"
                    )
                continue
            record.update({"status": "complete", "finished_at": utc_now()})
            self.state.save()
            print(f"DONE {job.job_id}: {output_dataset}")
            return

    def run(self, *, only_job: str | None = None, max_jobs: int | None = None) -> None:
        self.validate()
        jobs = build_jobs(self.config)
        selected = [job for job in jobs if only_job is None or job.job_id == only_job]
        if only_job and not selected:
            raise ValueError(f"Unknown job {only_job!r}")
        if max_jobs is not None:
            selected = selected[:max_jobs]
        print(f"Planned jobs: {len(selected)}")
        for job in selected:
            self.run_job(job)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "kaggle_pipeline.yaml",
        help="YAML pipeline matrix and Kaggle settings.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without calling Kaggle.")
    parser.add_argument("--only-job", help="Run one exact job ID from --dry-run output.")
    parser.add_argument("--max-jobs", type=int, help="Run at most this many planned jobs.")
    parser.add_argument("--kaggle-cli", default=None, help="Override the Kaggle executable path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.kaggle_cli:
        config.setdefault("kaggle", {})["cli"] = args.kaggle_cli
    if args.dry_run:
        for job in build_jobs(config):
            print(job.job_id)
        return 0
    runner = PipelineRunner(config, config_path)
    runner.run(only_job=args.only_job, max_jobs=args.max_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
