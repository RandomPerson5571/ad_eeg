"""Dataset-level preprocessing QC reports from subject logs (no reprocessing)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg.config import load_experiment
from eeg.io import write_json
from eeg.paths import qc_report_dir, subject_log_path
from eeg.qc import BAND_RANGES, backfill_spectral_qc, log_to_summary_row
from eeg.repro import preprocessing_fingerprint

DISTRIBUTION_METRICS = [
    "pct_epochs_rejected",
    "ica_n_removed",
    "ica_n_components_fitted",
    "runtime_seconds",
    "snr_db",
    "n_bad_channels",
    "ica_eeg_rank",
] + [f"{b}_delta_uv2" for b in BAND_RANGES]


def load_subject_logs(dataset_name: str, experiment: str) -> list[dict[str, Any]]:
    logs_dir = subject_log_path(dataset_name, experiment, "sub-000").parent
    if not logs_dir.exists():
        return []
    logs = []
    for path in sorted(logs_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            logs.append(json.load(f))
    return logs


def compute_distribution(values: list[float], n_bins: int = 15) -> dict[str, Any]:
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "iqr": [None, None],
            "min": None,
            "max": None,
            "histogram": {"bins": [], "counts": []},
        }
    q25, q75 = np.percentile(arr, [25, 75])
    counts, bin_edges = np.histogram(arr, bins=min(n_bins, max(5, arr.size)))
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "iqr": [round(float(q25), 4), round(float(q75), 4)],
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "histogram": {
            "bins": [round(float(b), 4) for b in bin_edges.tolist()],
            "counts": counts.tolist(),
        },
    }


def detect_outliers(df: pd.DataFrame) -> list[dict[str, str]]:
    outliers: list[dict[str, str]] = []
    if "pct_epochs_rejected" in df.columns:
        col = pd.to_numeric(df["pct_epochs_rejected"], errors="coerce").dropna()
        if len(col) > 2:
            mean, std = col.mean(), col.std()
            if std > 0:
                for _, row in df.iterrows():
                    val = row.get("pct_epochs_rejected")
                    if val is not None and float(val) > mean + 2 * std:
                        outliers.append(
                            {
                                "participant_id": str(row["participant_id"]),
                                "reason": f"rejection {val}% > mean+2σ ({mean + 2 * std:.1f}%)",
                            }
                        )
    for _, row in df.iterrows():
        fitted = row.get("ica_n_components_fitted")
        removed = row.get("ica_n_removed")
        if fitted is not None and removed is not None:
            try:
                if int(removed) >= int(fitted) and int(fitted) > 0:
                    outliers.append(
                        {
                            "participant_id": str(row["participant_id"]),
                            "reason": "ICA removed all fitted components",
                        }
                    )
            except (TypeError, ValueError):
                pass
    return outliers


def build_summary_dataframe(logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [log_to_summary_row(log) for log in logs]
    return pd.DataFrame(rows)


def build_summary_json(
    df: pd.DataFrame,
    dataset_name: str,
    experiment: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    distributions: dict[str, Any] = {}
    for metric in DISTRIBUTION_METRICS:
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[df["status"] == "ok"][metric], errors="coerce").dropna().tolist()
        distributions[metric] = compute_distribution(vals)

    failures = []
    if "status" in df.columns:
        for _, row in df[df["status"] == "error"].iterrows():
            failures.append(
                {
                    "participant_id": row.get("participant_id"),
                    "error": row.get("error"),
                    "error_stage": row.get("error_stage"),
                }
            )

    return {
        "dataset": dataset_name,
        "experiment": experiment,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": preprocessing_fingerprint(config),
        "n_subjects": {
            "total": len(df),
            "ok": int(status_counts.get("ok", 0)),
            "failed": int(status_counts.get("error", 0)),
            "skipped": int(status_counts.get("skipped", 0)),
        },
        "distributions": distributions,
        "outliers": detect_outliers(df),
        "failures": failures,
    }


def _ascii_histogram(hist: dict[str, Any], width: int = 40) -> str:
    counts = hist.get("counts", [])
    bins = hist.get("bins", [])
    if not counts or not bins:
        return "(no data)"
    max_count = max(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        bar = "#" * int(width * count / max_count)
        lo = bins[i] if i < len(bins) else ""
        hi = bins[i + 1] if i + 1 < len(bins) else ""
        lines.append(f"  [{lo:.2g}, {hi:.2g}): {bar} ({count})")
    return "\n".join(lines)


def render_dataset_report_md(summary: dict[str, Any], df: pd.DataFrame) -> str:
    fp = summary.get("fingerprint", {})
    lines = [
        "# Preprocessing Dataset Report",
        "",
        f"**Dataset:** {summary.get('dataset')}  ",
        f"**Experiment:** {summary.get('experiment')}  ",
        f"**Generated:** {summary.get('generated_at')}  ",
        "",
        "## Fingerprint",
        "",
        "| Key | Value |",
        "|-----|-------|",
    ]
    for key in (
        "mne_version",
        "autoreject_version",
        "asrpy_version",
        "git_commit",
        "config_sha256",
        "python_version",
    ):
        lines.append(f"| {key} | {fp.get(key, 'n/a')} |")

    ns = summary.get("n_subjects", {})
    lines.extend(
        [
            "",
            "## Subjects",
            "",
            f"- OK: {ns.get('ok', 0)}",
            f"- Failed: {ns.get('failed', 0)}",
            f"- Skipped: {ns.get('skipped', 0)}",
            "",
        ]
    )

    failures = summary.get("failures", [])
    if failures:
        lines.append("### Failures")
        lines.append("")
        for f in failures:
            lines.append(
                f"- **{f.get('participant_id')}** ({f.get('error_stage')}): {f.get('error')}"
            )
        lines.append("")

    lines.extend(["## Metric Distributions", ""])
    for metric, dist in summary.get("distributions", {}).items():
        if dist.get("count", 0) == 0:
            continue
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| Stat | Value |")
        lines.append("|------|-------|")
        for stat in ("mean", "median", "std", "min", "max"):
            lines.append(f"| {stat} | {dist.get(stat)} |")
        iqr = dist.get("iqr", [None, None])
        lines.append(f"| IQR | [{iqr[0]}, {iqr[1]}] |")
        lines.append("")
        lines.append("```")
        lines.append(_ascii_histogram(dist.get("histogram", {})))
        lines.append("```")
        lines.append("")

    outliers = summary.get("outliers", [])
    if outliers:
        lines.append("## Outliers")
        lines.append("")
        for o in outliers:
            lines.append(f"- **{o['participant_id']}**: {o['reason']}")
        lines.append("")

    return "\n".join(lines)


def render_html_dashboard(df: pd.DataFrame, out_dir: Path) -> str:
    """Generate qc/index.html with sortable subject table."""
    rows_html = []
    for _, row in df.iterrows():
        pid = row.get("participant_id", "")
        png = f"{pid}.png"
        png_path = out_dir / png
        img = (
            f'<a href="{png}"><img src="{png}" alt="{pid}" style="height:48px"></a>'
            if png_path.exists()
            else "—"
        )
        rows_html.append(
            f"<tr>"
            f'<td><a href="{png}">{pid}</a></td>'
            f"<td>{row.get('pct_epochs_rejected', '')}</td>"
            f"<td>{row.get('ica_n_removed', '')}</td>"
            f"<td>{row.get('ica_eeg_rank', '')}</td>"
            f"<td>{row.get('runtime_seconds', '')}</td>"
            f"<td>{row.get('status', '')}</td>"
            f"<td>{img}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Preprocessing QC Dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
th {{ background: #f4f4f4; cursor: pointer; }}
tr:hover {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Preprocessing QC Dashboard</h1>
<p>Generated {datetime.now(timezone.utc).isoformat()}</p>
<table id="qc-table">
<thead>
<tr>
<th>Subject</th><th>Epoch rejection %</th><th>ICA removed</th>
<th>Rank</th><th>Runtime (s)</th><th>Status</th><th>QC image</th>
</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
<script>
document.querySelectorAll('#qc-table th').forEach((th, idx) => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const rows = [...table.querySelectorAll('tbody tr')];
    const asc = th.dataset.sort !== 'asc';
    table.querySelectorAll('th').forEach(h => delete h.dataset.sort);
    th.dataset.sort = asc ? 'asc' : 'desc';
    rows.sort((a, b) => {{
      const av = a.children[idx].textContent.trim();
      const bv = b.children[idx].textContent.trim();
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
      return asc ? cmp : -cmp;
    }});
    rows.forEach(r => table.querySelector('tbody').appendChild(r));
  }});
}});
</script>
</body>
</html>"""
    return html


def write_preprocess_report(
    dataset_name: str,
    experiment: str,
    config: dict[str, Any] | None = None,
    qc_plots: bool = False,
    dataset_spec=None,
) -> dict[str, Path]:
    """Write summary.csv, summary.json, dataset_report.md, and optional HTML dashboard."""
    if config is None:
        config = load_experiment(experiment)

    out_dir = qc_report_dir(dataset_name, experiment)
    out_dir.mkdir(parents=True, exist_ok=True)

    backfill_spectral_qc(dataset_name, experiment)
    logs = load_subject_logs(dataset_name, experiment)
    df = build_summary_dataframe(logs)
    summary = build_summary_json(df, dataset_name, experiment, config)

    paths: dict[str, Path] = {
        "summary_csv": out_dir / "summary.csv",
        "summary_json": out_dir / "summary.json",
        "dataset_report_md": out_dir / "dataset_report.md",
    }
    df.to_csv(paths["summary_csv"], index=False)
    write_json(paths["summary_json"], summary)
    paths["dataset_report_md"].write_text(
        render_dataset_report_md(summary, df), encoding="utf-8"
    )

    html = render_html_dashboard(df, out_dir)
    paths["index_html"] = out_dir / "index.html"
    paths["index_html"].write_text(html, encoding="utf-8")

    if qc_plots and dataset_spec is not None:
        from eeg.visualization import plot_preprocessing_panels_from_checkpoints

        for log in logs:
            if log.get("status") not in ("ok", "skipped"):
                continue
            pid = log.get("participant_id", "")
            try:
                snum = int(str(pid).replace("sub-", ""))
            except ValueError:
                continue
            plot_preprocessing_panels_from_checkpoints(
                dataset_spec, snum, experiment, output_dir=out_dir
            )

    return paths


def compare_experiments(
    dataset_name: str,
    experiments: list[str],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Compare cohort metrics across experiments (reads existing summary.json files)."""
    compare_metrics = [
        ("pct_epochs_rejected", "Epoch rejection % (median)"),
        ("runtime_seconds", "Runtime (median)"),
        ("ica_n_removed", "ICA removed (median)"),
        ("alpha_delta_uv2", "Mean alpha Δ (µV²)"),
    ]
    rows = []
    for metric_key, label in compare_metrics:
        row: dict[str, Any] = {"metric": label}
        for exp in experiments:
            summary_path = qc_report_dir(dataset_name, exp) / "summary.json"
            if not summary_path.exists():
                row[exp] = None
                continue
            with summary_path.open(encoding="utf-8") as f:
                summary = json.load(f)
            dist = summary.get("distributions", {}).get(metric_key, {})
            if metric_key == "alpha_delta_uv2":
                row[exp] = dist.get("mean")
            else:
                row[exp] = dist.get("median")
        rows.append(row)
    df = pd.DataFrame(rows)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        md_lines = ["# Experiment Comparison", ""]
        md_lines.append("| " + " | ".join(df.columns) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            md_lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
        output_path.with_suffix(".md").write_text("\n".join(md_lines), encoding="utf-8")
    return df
