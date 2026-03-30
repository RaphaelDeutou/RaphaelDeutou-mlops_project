# === FICHIER : src/train.py (VERSION CORRIGÉE - MLflow run centralisé) ===
import argparse
import yaml
import pandas as pd
from pathlib import Path
import mlflow
from src.data.download_data import download_creditcard_data
from src.preprocessing.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer
from src.evaluation.evaluator import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    mlflow.set_tracking_uri(config["training"]["tracking_uri"])
    mlflow.set_experiment(config["training"]["experiment_name"])
    
    # === UN SEUL run MLflow pour tout le pipeline ===
    with mlflow.start_run(run_name="xgboost_run") as run:
        print(f"🏃 Starting MLflow run: {run.info.run_id}")
        
        # Download + preprocess
        download_creditcard_data()
        df = pd.read_csv(config["data"]["raw_path"])
        
        preprocessor = Preprocessor(scale=config["preprocessing"]["scale_features"])
        df = preprocessor.fit_transform(df)
        
        X = df.drop("Class", axis=1)
        y = df["Class"]
        
        # Split stratifié
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=config["preprocessing"]["test_size"], 
            stratify=y, random_state=config["preprocessing"]["random_state"]
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, 
            random_state=config["preprocessing"]["random_state"]
        )
        
        # Training
        trainer = ModelTrainer("xgboost")
        model = trainer.train(X_train, y_train, config["models"]["xgboost"])
        
        # Evaluation (dans le même run)
        evaluate_model(model, X_test, y_test, run.info.run_id)
        
        print("✅ Entraînement terminé avec succès - Tout est loggé dans MLflow !")
        print(f"🔗 View run at: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}")

if __name__ == "__main__":
    main()