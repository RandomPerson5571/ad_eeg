import argparse
import json
import os

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifier_models.train_utils import (
    classification_metrics,
    prepare_training_data,
    save_metrics,
    save_model_artifact,
)
from config import RANDOM_STATE, RESULTS_DIR, TEST_SIZE

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_FILE = os.path.join(MODEL_DIR, "eeg_mlp_classifier.joblib")


def build_pipeline(hidden_layers=(128, 64), random_state=RANDOM_STATE):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    tol=1e-4,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


def train_mlp(test_size=TEST_SIZE, hidden_layers=(128, 64), random_state=RANDOM_STATE, output_path=MODEL_FILE):
    x_train, x_test, y_train, y_test, label_encoder, split_ids = prepare_training_data(
        test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(hidden_layers=hidden_layers, random_state=random_state)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = classification_metrics(y_test, y_pred, label_encoder)
    metrics["model"] = "mlp"
    metrics["split_ids"] = split_ids

    feature_names = list(x_train.columns)
    save_model_artifact(
        pipeline,
        label_encoder,
        feature_names,
        split_ids,
        "mlp",
        output_path,
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    splits_path = os.path.join(RESULTS_DIR, "subject_splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(split_ids, f, indent=2)

    save_metrics(metrics, os.path.join(RESULTS_DIR, "metrics_mlp.json"))

    return pipeline, label_encoder, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP classifier on extracted EEG features.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Fraction of data to reserve for testing.")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes.")
    parser.add_argument("--output", type=str, default=MODEL_FILE, help="Path to save the trained model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_mlp(
        test_size=args.test_size,
        hidden_layers=tuple(args.hidden_layers),
        output_path=args.output,
    )
