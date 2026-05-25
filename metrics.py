import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from train import trained_models, model_encoder
from feature_engineering import split_features
from data_cleaning import apply_label, clean_data
from dataloading import load_data


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # AUC (only if model supports probabilities)
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class='ovr',
            average='macro'
        )
    except:
        auc = None

    cm = confusion_matrix(y_test, y_pred)

    # Specificity calculation (per class)
    specificities = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        specificities.append(tn / (tn + fp))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro'),
        "auc": auc,
        "specificity": np.mean(specificities)
    }


def evaluate_all_models(models_dict, X_test, y_test):

    results = []

    for name, model in models_dict.items():

        metrics = evaluate_model(model, X_test, y_test)

        metrics["model"] = name
        results.append(metrics)

    df_results = pd.DataFrame(results)

    # reorder columns nicely
    df_results = df_results[
        ["model", "accuracy", "precision", "recall", "f1", "auc", "specificity"]
    ]

    return df_results

cleaned_dataset = clean_data(apply_label(load_data()))

data_X_test = split_features(cleaned_dataset)["X_test"]
data_y_test = split_features(cleaned_dataset)["y_test"]

test_data = model_encoder.transform(data_y_test)

selected_models = trained_models  # dictionary of models

df_results = evaluate_all_models(
    selected_models,
    data_X_test,
    test_data
)

print(df_results)