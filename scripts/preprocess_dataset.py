#!/usr/bin/env python3
"""Preprocess raw EEG into staged checkpoints
(raw → filtered → ica → clean → epochs).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from eeg.cli import (
    add_common_args,
    resolve_datasets_arg,
    subject_num_from_id,
)
from eeg.config import config_fingerprint, load_experiment
from eeg.io import list_subjects
from eeg.paths import preprocessed_dir, raw_eeg_path
from eeg.preprocess_report import write_preprocess_report
from eeg.preprocessing import preprocess_subject
from eeg.runner import (
    run_parallel,
    summarize_batch,
    update_experiment_metadata,
)


def _worker(
    raw_path,
    dataset_name,
    experiment,
    participant_id,
    config,
    config_fp,
    force,
    output_dir,
):
    return preprocess_subject(
        raw_path=Path(raw_path),
        participant_id=participant_id,
        dataset_name=dataset_name,
        experiment=experiment,
        config=config,
        config_fp=config_fp,
        force=force,
        output_dir=output_dir,
    )


def run_preprocess(
    dataset_tag: str,
    experiment: str,
    workers: int = 1,
    force: bool = False,
    limit: int | None = None,
    subject: str | None = None,
    qc_plots: bool = False,
    output_dir: str | Path | None = None,
):
    """
    Preprocess EEG data.

    If output_dir is provided, it replaces ``data/preprocessed`` as the
    checkpoint root. Each dataset/experiment keeps the same filenames as the
    standard pipeline.

    Example:

        /kaggle/working/pipeline_output/
            dataset_name/
                experiment/
                    sub-001_epo.fif
                    sub-002_epo.fif
                    ...
    """

    config = load_experiment(experiment)
    config_fp = config_fingerprint(config)

    # Convert to Path once.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    datasets = resolve_datasets_arg(dataset_tag)

    all_results = []

    for ds in datasets:

        config = load_experiment(experiment)
        config_fp = config_fingerprint(config)

        # ----------------------------------------------------
        # Determine where this dataset's preprocessed EEG
        # should be stored.
        # ----------------------------------------------------

        if output_dir is not None:
            dataset_output_dir = (
                output_dir
                / ds.name
                / experiment
            )

            dataset_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
        else:
            # Preserve the original behavior when no
            # output_dir is provided.
            dataset_output_dir = preprocessed_dir(
                ds.name,
                experiment,
            )

            dataset_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        # ``None`` is significant: it tells preprocess_subject to use the
        # canonical data/preprocessed/{dataset}/{experiment} paths. Passing
        # dataset_output_dir in the default case used to select the custom
        # layout and duplicate the dataset name.
        worker_output_dir = (
            str(dataset_output_dir) if output_dir is not None else None
        )

        participants = list_subjects(ds)
        participant_index = participants[["participant_id", "Group"]].to_dict(
            orient="records"
        )

        if subject:
            participants = participants[
                participants["participant_id"] == subject
            ]

        tasks = []

        for idx, row in participants.iterrows():

            subject_num = idx + 1

            participant_id = row["participant_id"]

            raw_path = str(
                raw_eeg_path(
                    ds,
                    subject_num,
                )
            )

            tasks.append(
                (
                    raw_path,
                    ds.name,
                    experiment,
                    participant_id,
                    config,
                    config_fp,
                    force,
                    worker_output_dir,
                )
            )

        if limit:
            tasks = tasks[:limit]

        results = run_parallel(
            tasks,
            _worker,
            workers=workers,
        )

        batch = summarize_batch(results)

        update_experiment_metadata(
            ds.name,
            experiment,
            config,
            "preprocessed",
            batch,
            len(tasks),
            extra_metadata={"participant_index": participant_index},
        )

        report_paths = write_preprocess_report(
            ds.name,
            experiment,
            config=config,
            qc_plots=qc_plots,
            dataset_spec=ds,
        )

        print(
            f"[{ds.name}] "
            f"completed={batch.completed} "
            f"skipped={batch.skipped} "
            f"failed={batch.failed}"
        )

        print(
            f"[{ds.name}] "
            f"output: {dataset_output_dir}"
        )

        print(
            f"[{ds.name}] "
            f"QC report: {report_paths['summary_csv']}"
        )

        all_results.extend(results)

    return all_results


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw EEG "
            "with staged checkpoints."
        )
    )

    add_common_args(parser)

    parser.add_argument(
        "--subject",
        help="Single subject: sub-001 or 1",
    )

    parser.add_argument(
        "--qc-plots",
        action="store_true",
        help=(
            "Generate per-subject QC PNGs "
            "from checkpoints after preprocessing"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where preprocessed EEG "
            "checkpoints will be saved."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    subject = None

    if args.subject:

        from eeg.cli import parse_subject_arg

        subject = parse_subject_arg(
            args.subject
        )

    run_preprocess(
        dataset_tag=args.dataset,
        experiment=args.experiment,
        workers=args.workers,
        force=args.force,
        limit=args.limit,
        subject=subject,
        qc_plots=args.qc_plots,
        output_dir=args.output_dir,
    )
