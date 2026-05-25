import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a single trained model.
    Returns all major evaluation metrics.
    """

    y_pred = model.predict(X_test)

    try:

        y_prob = model.predict_proba(X_test)

        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

    except Exception:

        auc = None

    cm = confusion_matrix(y_test, y_pred)

    specificities = []

    for i in range(len(cm)):

        tp = cm[i, i]

        fn = np.sum(cm[i, :]) - tp

        fp = np.sum(cm[:, i]) - tp

        tn = np.sum(cm) - (tp + fn + fp)

        specificity = tn / (tn + fp)

        specificities.append(specificity)

    mean_specificity = np.mean(specificities)

    return {

        "accuracy": accuracy_score(y_test, y_pred),

        "precision": precision_score(
            y_test,
            y_pred,
            average="macro"
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            average="macro"
        ),

        "f1": f1_score(
            y_test,
            y_pred,
            average="macro"
        ),

        "auc": auc,

        "specificity": mean_specificity,

        # optional extras
        "confusion_matrix": cm,

        "classification_report": classification_report(
            y_test,
            y_pred
        )
    }

def evaluate_all_models(models_dict, X_test, y_test):
    """
    Evaluates all trained models
    and returns a comparison DataFrame.
    """

    results = []

    for model_name, model in models_dict.items():

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        # attach model name
        metrics["model"] = model_name

        results.append(metrics)

    df_results = pd.DataFrame(results)
    
    metric_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "specificity"
    ]

    return df_results[metric_columns]