import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    }


def top2_group_indices(probabilities):
    probabilities = np.asarray(probabilities)
    return np.argsort(probabilities, axis=1)[:, -2:][:, ::-1]
