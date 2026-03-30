# === FICHIER : src/models/trainer.py (VERSION CORRIGÉE - sans with start_run) ===
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

class ModelTrainer:
    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self.model = None

    def train(self, X_train, y_train, params: dict):
        if self.model_name == "logistic_regression":
            self.model = LogisticRegression(**params)
        elif self.model_name == "random_forest":
            self.model = RandomForestClassifier(**params)
        elif self.model_name == "xgboost":
            self.model = XGBClassifier(**params, eval_metric="aucpr")
        
        # SMOTE + entraînement
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        self.model.fit(X_res, y_res)
        
        # Logging du modèle (le run est déjà ouvert dans train.py)
        mlflow.sklearn.log_model(self.model, "model")
        joblib.dump(self.model, f"models/{self.model_name}.joblib")
        
        print(f"✅ Modèle {self.model_name} entraîné et loggé")
        return self.model