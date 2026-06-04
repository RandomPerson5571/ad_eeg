import argparse
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier, plot_importance
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from util.io import load_features
from config import TEST_SIZE, RANDOM_STATE

MODEL_DIR = "classifier_models/saved_models"
MODEL_FILE = "classifier_models/saved_models/xgboost_eeg_classifier.joblib"


def train_xgboost(test_size=TEST_SIZE, random_state=RANDOM_STATE, n_iter=10, cv=3):
    features, labels = load_features()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    estimators = [
        ("encoder", TargetEncoder()),
        ("clf", XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss")),
    ]

    pipe = Pipeline(steps=estimators)
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
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )

    opt.fit(x_train, y_train)

    predictions = opt.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    print("XGBoost model training complete")
    print(f"Test ROC AUC: {opt.score(x_test, y_test):.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Classification report:\n")
    print(report)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(opt.best_estimator_, MODEL_FILE)
    print(f"Saved XGBoost model to {MODEL_FILE}")

    xgboost_step = opt.best_estimator_.named_steps["clf"]
    print("Top 10 feature importances:")
    importances = xgboost_step.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices[:10]:
        print(f"Feature {i}: importance = {importances[i]:.4f}")

    return opt, x_test, y_test


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EEG XGBoost classifier using extracted features.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test set fraction.")
    parser.add_argument("--n-iter", type=int, default=10, help="BayesSearchCV iterations.")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_xgboost(test_size=args.test_size, n_iter=args.n_iter, cv=args.cv)
