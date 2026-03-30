# src/api/main.py - FastAPI inference service
from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel
import logging
from datetime import datetime

app = FastAPI(title="MLOps Fraud Detection API")

# Chargement du dernier modèle MLflow
model_uri = "models:/credit_fraud_detection/latest"  # ou via run_id
model = mlflow.sklearn.load_model(model_uri)

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: PredictionRequest):
    df = pd.DataFrame([req.features], columns=[f"V{i}" for i in range(1,29)] + ["Time", "Amount"])
    prob = model.predict_proba(df)[0][1]
    
    # Logging chaque requête (observabilité)
    logging.info({
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": "latest",
        "fraud_probability": float(prob),
        "input_hash": hash(str(req.features))
    })
    
    return {"fraud_probability": float(prob), "is_fraud": prob > 0.5}

@app.get("/health")
def health():
    return {"status": "healthy"}