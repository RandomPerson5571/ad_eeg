"""Add deprecation banner to legacy notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

banner = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["> **Deprecated:** Use `notebooks/kaggle/` (00-08) instead.\n"],
}

for nb in [
    "01_dataset_overview.ipynb",
    "02_preprocessing_and_qc.ipynb",
    "03_feature_extraction.ipynb",
    "04_model_training.ipynb",
    "05_results_and_interpretation.ipynb",
]:
    p = ROOT / "notebooks" / nb
    data = json.loads(p.read_text(encoding="utf-8"))
    if data["cells"] and "Deprecated" in "".join(data["cells"][0].get("source", [])):
        continue
    data["cells"].insert(0, banner)
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("updated", nb)
