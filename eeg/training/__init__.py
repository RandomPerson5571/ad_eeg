"""Training package — benchmark, train, datasets, evaluation."""

from eeg.training.benchmark import MODEL_REGISTRY, run_benchmark
from eeg.training.datasets import (
    feature_columns,
    load_selected_features,
    prepare_data,
    subject_level_split,
    validate_feature_schema,
)
from eeg.training.evaluation import (
    bootstrap_ci,
    compute_benchmark_metrics,
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
    plot_pr,
    plot_roc,
    plot_runtime_bar,
)
from eeg.training.inference import predict_with_timing
from eeg.training.train import (
    classification_metrics,
    save_model_artifact,
    train_mlp,
    train_models,
    train_xgboost,
)

__all__ = [
    "MODEL_REGISTRY",
    "run_benchmark",
    "feature_columns",
    "load_selected_features",
    "prepare_data",
    "subject_level_split",
    "validate_feature_schema",
    "bootstrap_ci",
    "compute_benchmark_metrics",
    "plot_calibration",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_learning_curve",
    "plot_pr",
    "plot_roc",
    "plot_runtime_bar",
    "predict_with_timing",
    "classification_metrics",
    "save_model_artifact",
    "train_mlp",
    "train_models",
    "train_xgboost",
]
