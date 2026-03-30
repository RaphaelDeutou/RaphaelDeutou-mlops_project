# src/monitoring/drift_simulation.py - Simulation data/concept drift + alerte MLflow
import pandas as pd
import mlflow
import numpy as np
from datetime import datetime

def simulate_drift():
    df = pd.read_csv("data/raw/creditcard.csv")
    # Data drift : Amount * 1.3 sur 20% des lignes
    mask = np.random.rand(len(df)) < 0.2
    df.loc[mask, "Amount"] = df.loc[mask, "Amount"] * 1.3
    df.loc[mask, "V1"] = df.loc[mask, "V1"] * 0.8  # concept drift simulé
    
    # Log dans MLflow comme run de monitoring
    with mlflow.start_run(run_name="drift_simulation", tags={"type": "monitoring"}):
        mlflow.log_metric("drift_amount_factor", 1.3)
        mlflow.log_metric("drift_affected_rows_pct", 20.0)
        mlflow.log_param("drift_type", "data + concept")
        mlflow.log_artifact("data/raw/creditcard.csv", artifact_path="drifted_data")
        
        print("🚨 ALERTE : Data/Concept drift détecté et loggé dans MLflow !")
        # En production : envoi email/Slack (simulation ici)
    return "Drift simulation terminée"

if __name__ == "__main__":
    simulate_drift()