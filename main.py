# =========================
# MAIN ORCHESTRATION FILE
# =========================

from dataloading import load_data
from data_cleaning import clean_data, apply_label
from feature_engineering import split_features, get_feature_groups
from data_preprocessing import build_preprocessor, encode_target
from models import get_models
from training import run_batch_grid_search
from evaluation import evaluate_all_models

import pandas as pd


# =========================
# 1. LOAD + CLEAN DATA
# =========================

raw_data = load_data()
cleaned_data = clean_data(apply_label(raw_data))


# =========================
# 2. FEATURE SPLIT
# =========================

features = split_features(cleaned_data)

X_train = features["X_train"]
X_test = features["X_test"]
y_train = features["y_train"]
y_test = features["y_test"]


# =========================
# 3. ENCODE TARGET
# =========================

y_train_enc, encoder = encode_target(y_train)
y_test_enc = encoder.transform(y_test)


# =========================
# 4. PREPROCESSING PIPELINE
# =========================

categorical, numerical = get_feature_groups(X_train)
preprocessor = build_preprocessor(categorical, numerical)


# =========================
# 5. MODEL REGISTRY
# =========================

models = get_models()
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


# =========================
# 6. TRAIN + TUNE MODELS
# =========================

results = run_batch_grid_search(
    preprocessor,
    models,
    param_grids,
    X_train,
    y_train_enc
)


# =========================
# 7. EVALUATE BEST MODELS
# =========================

final_results_df = evaluate_all_models(
    {name: res["best_estimator"] for name, res in results.items()},
    X_test,
    y_test_enc
)


# =========================
# 8. OUTPUT RESULTS
# =========================

print("\n===== MODEL PERFORMANCE TABLE =====\n")
print(final_results_df)

best_model_name = final_results_df.sort_values("f1", ascending=False).iloc[0]["model"]

print("\nBEST MODEL:", best_model_name)


# =========================
# 9. SAVE BEST MODEL (OPTIONAL BUT IMPORTANT)
# =========================

best_model = results[best_model_name]["best_estimator"]

import joblib
joblib.dump(best_model, "best_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")