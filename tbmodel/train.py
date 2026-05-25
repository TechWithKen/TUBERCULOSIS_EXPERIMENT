import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_model_registry():

    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, random_state=42),
            "params": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__solver": ["liblinear", "lbfgs"],
                "model__class_weight": [None, "balanced"]
            }
        },

        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
                "model__class_weight": [None, "balanced"]
            }
        },

        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"]
            }
        },

        "XGBoost": {
            "model": XGBClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1]
            }
        }
    }

def run_batch_grid_search(preprocessor, X_train, y_train, model_registry, cv=5):

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = {}

    for name, config in model_registry.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", config["model"])
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        results[name] = {
            "best_score": grid.best_score_,
            "best_params": grid.best_params_,
            "best_estimator": grid.best_estimator_
        }

    return results


def get_trained_models(grid_results, X_train, y_train):

    trained_models = {}

    for name, result in grid_results.items():

        model = result["best_estimator"]

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models