"""
Model training and evaluation for the Finland train-delay prediction project.

The script contains reusable versions of the models evaluated in
Finland_dataset_evaluation_MachineLearning.ipynb.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 1

FEATURE_COLUMNS = [
    "month",
    "day_of_week",
    "type_encoded",
    "scheduled_time_minutes",
    "Air temperature",
    "Wind speed",
    "Snow depth",
    "Horizontal visibility",
    "sin_time",
    "cos_time",
    "sin_month",
    "cos_month",
    "sin_day",
    "cos_day",
]


def load_processed_data(data_dir: str | Path = "../data/processed"):
    """Load the undersampled training data and untouched test data."""
    data_dir = Path(data_dir)

    train_df = pd.read_csv(data_dir / "train_undersampled.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["delayed"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["delayed"]

    return X_train, y_train, X_test, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Fit a classifier and return accuracy and ROC-AUC."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with the same scaling approach as the notebook."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        random_state=SEED,
        max_iter=1000,
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    return model, {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_hist_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train the baseline HistGradientBoosting classifier."""
    model = HistGradientBoostingClassifier(random_state=SEED)
    return model, evaluate_model(
        model, X_train, y_train, X_test, y_test
    )


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train the baseline Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1,
    )
    return model, evaluate_model(
        model, X_train, y_train, X_test, y_test
    )


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train the baseline XGBoost classifier."""
    model = XGBClassifier(
        random_state=SEED,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=2,
    )
    return model, evaluate_model(
        model, X_train, y_train, X_test, y_test
    )


def tune_xgboost(X_train, y_train, sample_size=300_000):
    """Run the randomized XGBoost search used in the notebook."""
    X_sample = X_train.sample(n=sample_size, random_state=SEED)
    y_sample = y_train.loc[X_sample.index]

    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=SEED,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=2,
        ),
        param_distributions=params,
        n_iter=15,
        cv=3,
        scoring="roc_auc",
        random_state=SEED,
        n_jobs=1,
    )

    search.fit(X_sample, y_sample)
    return search


def train_final_xgboost(X_train, y_train, X_test, y_test):
    """Train the focused XGBoost configuration used for final evaluation."""
    model = XGBClassifier(
        random_state=SEED,
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        min_child_weight=1,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=2,
    )

    return model, evaluate_model(
        model, X_train, y_train, X_test, y_test
    )


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_processed_data()

    results = []

    for name, trainer in [
        ("Logistic Regression", train_logistic_regression),
        ("HistGradientBoosting", train_hist_gradient_boosting),
        ("Random Forest", train_random_forest),
        ("Focused XGBoost", train_final_xgboost),
    ]:
        _, metrics = trainer(X_train, y_train, X_test, y_test)
        results.append({"Model": name, **metrics})

    print(pd.DataFrame(results))
