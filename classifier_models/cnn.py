import os
import argparse
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from util.io import load_features
from config import TEST_SIZE, RANDOM_STATE

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_FILE = os.path.join(MODEL_DIR, "eeg_mlp_classifier.joblib")


def prepare_data(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    features, labels = load_features()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=test_size,
        stratify=encoded_labels,
        random_state=random_state,
    )

    return x_train, x_test, y_train, y_test, label_encoder


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


def evaluate_model(pipeline, x_test, y_test, label_encoder):
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_model(pipeline, label_encoder, output_path=MODEL_FILE):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder}, output_path)
    print(f"Saved trained EEG classifier to {output_path}")


def train_eeg_model(test_size=TEST_SIZE, hidden_layers=(128, 64), random_state=RANDOM_STATE, output_path=MODEL_FILE):
    x_train, x_test, y_train, y_test, label_encoder = prepare_data(test_size=test_size, random_state=random_state)
    pipeline = build_pipeline(hidden_layers=hidden_layers, random_state=random_state)
    pipeline.fit(x_train, y_train)

    metrics = evaluate_model(pipeline, x_test, y_test, label_encoder)
    save_model(pipeline, label_encoder, output_path)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print("Classification report:\n")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    return pipeline, label_encoder, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EEG classification model using extracted features.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data to reserve for testing.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer sizes for the MLP classifier.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=MODEL_FILE,
        help="Path to save the trained model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_eeg_model(
        test_size=args.test_size,
        hidden_layers=tuple(args.hidden_layers),
        output_path=args.output,
    )
