import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except:
        auc = None

    cm = confusion_matrix(y_test, y_pred)

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
        "specificity": np.mean(specificities),
        "confusion_matrix": cm,
        "report": classification_report(y_test, y_pred)
    }