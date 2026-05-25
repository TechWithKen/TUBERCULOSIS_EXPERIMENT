import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from models import get_models
from data_preprocessing import build_preprocessor, encode_target
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from feature_engineering import split_features, get_feature_groups
from data_cleaning import apply_label, clean_data
from dataloading import load_data


def get_model_registry():
    """
    Returns models AND their hyperparameter search spaces.
    This is the key abstraction for batch tuning.
    """
    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"],
            "model__class_weight": [None, "balanced"]
        },

        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced"]
        },

        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        },

        "XGBoost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1]
        }
    }

    return param_grids



def run_batch_grid_search(preprocessor, models, param_grids, X_train, y_train, cv=5):
    """
    Runs GridSearchCV for ALL models and returns best results.
    """

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = {}

    for name in models.keys():

        print(f"\nTuning: {name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", models[name])
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
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




data_models = get_models()
data_param = get_model_registry()
cleaned_dataset = clean_data(apply_label(load_data()))
data_X_train = split_features(cleaned_dataset)["X_train"]
data_y_train = split_features(cleaned_dataset)["y_train"]


encoded_y_train, model_encoder = encode_target(data_y_train)

cat_num_features = get_feature_groups(data_X_train)
categorical, numerical = cat_num_features
data_preprocessor = build_preprocessor(categorical, numerical)



models_results = run_batch_grid_search(data_preprocessor, data_models, data_param, data_X_train, encoded_y_train)

best_log_reg_params = models_results["Logistic Regression"]["best_params"]
best_rf_params = models_results["Random Forest"]["best_params"]
best_svm_params = models_results["SVM"]["best_params"]
best_xgb_params = models_results["XGBoost"]["best_params"]


def best_param_models():

    log_reg = LogisticRegression(
        C=best_log_reg_params["model__C"],
        solver=best_log_reg_params["model__solver"],
        class_weight=best_log_reg_params["model__class_weight"],
        max_iter=2000,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=best_rf_params["model__n_estimators"],
        max_depth=best_rf_params["model__max_depth"],
        min_samples_split=best_rf_params["model__min_samples_split"],
        class_weight=best_rf_params["model__class_weight"],
        random_state=42
    )

    svm = SVC(
        C=best_svm_params["model__C"],
        kernel=best_svm_params["model__kernel"],
        probability=True,
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=best_xgb_params["model__n_estimators"],
        max_depth=best_xgb_params["model__max_depth"],
        learning_rate=best_xgb_params["model__learning_rate"],
        random_state=42
    )

    return log_reg, rf, svm, xgb


def ind_model_pipeline(preprocessor, X_train, y_train):

    best_parameterized_models = best_param_models()

    final_models = {}

    for model_name in best_parameterized_models:

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_name)
        ])

        pipeline.fit(X_train, y_train)

        # Store trained pipeline
        final_models[model_name] = pipeline

    return final_models


print(ind_model_pipeline(data_preprocessor, data_X_train, encoded_y_train))