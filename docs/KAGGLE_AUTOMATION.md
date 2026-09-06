# Automated Kaggle Pipeline

`scripts/run_kaggle_pipeline.py` runs the generated Kaggle notebooks one at a time.
It waits for the remote kernel, downloads the output, publishes a new version of a
cumulative Kaggle Dataset, and only then schedules the next notebook. Notebook 03
is automatically repeated when its output marker says `partial`.

## One-time setup

1. Install the project dependencies, including the Kaggle CLI:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Authenticate the CLI with `kaggle auth login`, `KAGGLE_API_TOKEN`, or the
   standard Kaggle credentials file.

3. Edit `configs/kaggle_pipeline.yaml`:

   - set `kaggle.raw_dataset` to the Dataset handle containing the raw
     `EEG_data/dataset2/` and `EEG_data/dataset3/` directories;
   - adjust `matrix.experiments` and `matrix.feature_sets` if needed;
   - use a Git branch containing the notebook/code changes in `repo_branch`.

4. Regenerate the notebooks after changing the generator:

   ```bash
   python scripts/generate_kaggle_notebooks.py
   ```

## Run and resume

Preview the exact serial plan:

```bash
python scripts/run_kaggle_pipeline.py --dry-run
```

Start or resume the workflow:

```bash
KAGGLE_USERNAME=your-user python scripts/run_kaggle_pipeline.py
```

State is saved to `.kaggle_pipeline/state.json`, which is ignored by Git. A
completed job is skipped on the next invocation. If a job fails, the runner stops,
keeps the rendered kernel and logs, and requires `retry_failed: true` before it is
retried.

The generated output Dataset handles are cumulative by dataset and feature set:

```text
owner/ad-eeg-pipeline-eyesclosed-full-preprocessed
owner/ad-eeg-pipeline-eyesclosed-full-epochs
owner/ad-eeg-pipeline-eyesclosed-full-features
owner/ad-eeg-pipeline-eyesclosed-full-selected
owner/ad-eeg-pipeline-eyesclosed-full-results
```

Each stage versions its existing Dataset instead of creating a new Dataset for
every retry. This preserves earlier experiment artifacts so later stages can
compare experiments and build the ablation/final benchmark outputs.

## Scheduling

The process itself is resumable, so it can be invoked by cron, macOS `launchd`,
or CI. For example, a daily cron entry can call a small wrapper that activates the
virtual environment and runs:

```text
python /absolute/path/to/ad_eeg/scripts/run_kaggle_pipeline.py
```

Use a persistent machine or CI runner for long jobs. The Kaggle kernel continues
remotely, while the next scheduler invocation can inspect the saved state and
continue from the last completed Dataset version.

## Starting from an existing artifact

To skip earlier stages, add a Dataset handle under `inputs` using the corresponding
family name (`preprocessed`, `epochs`, `features`, `selected`, or `results`) and
start the selected stage with `--only-job`. The normal full run does not need this;
the state file fills the handles automatically.
