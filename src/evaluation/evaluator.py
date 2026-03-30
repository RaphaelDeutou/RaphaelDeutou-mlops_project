# src/evaluation/evaluator.py
import mlflow
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(model, X_test, y_test, run_id: str):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "pr_auc": average_precision_score(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": (y_pred == y_test).mean(),  # placeholder, calcul réel ci-dessous
    }
    
    # Log artefacts
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.savefig("pr_curve.png")
    mlflow.log_artifact("pr_curve.png")
    
    mlflow.log_metrics(metrics)
    return metrics