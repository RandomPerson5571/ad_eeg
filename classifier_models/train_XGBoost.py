import argparse
import json
import os

import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from classifier_models.train_utils import (
    classification_metrics,
    prepare_training_data,
    results_path_for_dataset,
    save_metrics,
    save_model_artifact,
)
from config import DATASETS, RANDOM_STATE, RESULTS_DIR, TEST_SIZE

MODEL_DIR = os.path.join("classifier_models", "saved_models")


def model_file_for_dataset(dataset_id):
    return os.path.join(MODEL_DIR, f"xgboost_eeg_classifier_dataset{dataset_id}.joblib")


def train_xgboost(dataset_id, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_iter=10, cv=3):
    x_train, x_test, y_train, y_test, label_encoder, split_ids = prepare_training_data(
        dataset_id=dataset_id, test_size=test_size, random_state=random_state
    )

    pipe = Pipeline(
        steps=[
            ("encoder", TargetEncoder()),
            (
                "clf",
                XGBClassifier(
                    random_state=random_state,
                    eval_metric="mlogloss",
                    objective="multi:softprob",
                ),
            ),
        ]
    )

    search_space = {
        "clf__max_depth": Integer(2, 8),
        "clf__learning_rate": Real(0.001, 1.0, prior="log-uniform"),
        "clf__subsample": Real(0.5, 1.0),
        "clf__colsample_bytree": Real(0.5, 1.0),
        "clf__colsample_bylevel": Real(0.5, 1.0),
        "clf__colsample_bynode": Real(0.5, 1.0),
        "clf__reg_alpha": Real(0.0, 10.0),
        "clf__reg_lambda": Real(0.0, 10.0),
        "clf__gamma": Real(0.0, 10.0),
    }

    opt = BayesSearchCV(
        pipe,
        search_space,
        cv=cv,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )

    opt.fit(x_train, y_train)

    predictions = opt.predict(x_test)
    y_test_enc = label_encoder.transform(y_test)
    metrics = classification_metrics(y_test_enc, predictions, label_encoder)
    metrics["cv_best_score"] = float(opt.best_score_)
    metrics["model"] = "xgboost"
    metrics["dataset_id"] = dataset_id
    metrics["split_ids"] = split_ids

    print("XGBoost model training complete")
    print(f"CV balanced accuracy: {opt.best_score_:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")

    feature_names = list(x_train.columns)
    model_file = model_file_for_dataset(dataset_id)
    save_model_artifact(
        opt.best_estimator_,
        label_encoder,
        feature_names,
        split_ids,
        "xgboost",
        model_file,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    splits_path = results_path_for_dataset("subject_splits.json", dataset_id)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(split_ids, f, indent=2)

    xgboost_step = opt.best_estimator_.named_steps["clf"]
    importances = xgboost_step.feature_importances_
    indices = np.argsort(importances)[::-1]
    metrics["feature_importances"] = {
        feature_names[i]: float(importances[i]) for i in indices[:10]
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = results_path_for_dataset("metrics_xgboost.json", dataset_id)
    save_metrics(metrics, metrics_path)

    print("Top 10 feature importances:")
    for name, imp in metrics["feature_importances"].items():
        print(f"  {name}: {imp:.4f}")

    return opt, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EEG XGBoost classifier using extracted features.")
    parser.add_argument("--dataset", type=int, choices=DATASETS, required=True, help="Dataset ID to train on.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test set fraction.")
    parser.add_argument("--n-iter", type=int, default=10, help="BayesSearchCV iterations.")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_xgboost(dataset_id=args.dataset, test_size=args.test_size, n_iter=args.n_iter, cv=args.cv)
