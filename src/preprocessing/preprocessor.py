# === FICHIER : src/preprocessing/preprocessor.py (VERSION CORRIGÉE) ===
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class Preprocessor:
    def __init__(self, scale: bool = True):
        self.scaler = StandardScaler() if scale else None
        self.scale = scale

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Gestion NA et outliers (comme demandé dans l'énoncé)
        df = df.fillna(df.median())
        for col in df.columns:
            if col not in ['Time', 'Amount', 'Class']:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        
        # CRÉATION DES DOSSIERS (fix du bug actuel)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        if self.scale:
            numeric_cols = df.columns.drop(['Time', 'Amount', 'Class'], errors='ignore')
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            joblib.dump(self.scaler, "models/scaler.joblib")
            print("✅ Scaler sauvegardé dans models/scaler.joblib")
        
        df.to_csv("data/processed/creditcard_processed.csv", index=False)
        print("✅ Données prétraitées sauvegardées")
        return df