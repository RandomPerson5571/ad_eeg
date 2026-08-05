"""Backward-compatible entry point. Prefer: python pipeline.py"""
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.warn("main.py is deprecated; use pipeline.py", DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    sys.argv[1:1] = ["--stages", "preprocess,features"]
    from pipeline import parse_args
    from scripts.extract_features import run_extract
    from scripts.preprocess_dataset import run_preprocess

    args = parse_args()
    stages = {s.strip() for s in args.stages.split(",")}
    if "preprocess" in stages:
        run_preprocess(
            args.dataset, args.experiment, workers=args.workers, force=args.force, limit=args.limit
        )
    if "features" in stages:
        run_extract(
            args.dataset, args.experiment, workers=args.workers, force=args.force, limit=args.limit
        )
