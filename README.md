# Projet MLOps M2 - Détection de Fraude Cartes Bancaires

## Contexte

Solution MLOps complète pour industrialiser un modèle de détection de fraudes (UCI Credit Card Fraud).  
Reproductible sur machine vierge via Docker + MLflow + GitLab CI/CD complet.

## Structure du dépôt

### mlops-project/
### ├── data/raw/
### ├── src/
### │   ├── data/download_data.py
### │   ├── preprocessing/preprocessor.py
### │   ├── models/trainer.py
### │   ├── evaluation/evaluator.py
### │   ├── train.py
### │   ├── api/main.py
### │   └── monitoring/drift_simulation.py
### ├── docker/Dockerfile
### ├── docker-compose.yml
### ├── .gitlab-ci.yml
### ├── requirements.txt
### ├── config.yaml
### ├── README.md
### └── tests/
### ├── test_preprocessing.py
### └── test_model.py

### Commandes "one-command" (tout fonctionne en 1 ligne)

## 1. Cloner + installer
git clone https://github.com/RaphaelDeutou/RaphaelDeutou-mlops_project.git && cd mlops-project

## 2. Lancer TOUT (MLflow + API + entraînement)
docker compose up --build -d

## 3. Télécharger les données (une seule fois)
docker compose run --rm train python src/data/download_data.py

## 4. Lancer l'entraînement complet (avec MLflow tracking)
docker compose run --rm train python src/train.py --config config.yaml

## 5. Accéder à MLflow UI → http://localhost:5000

## 7. Simulation drift + alerte
docker compose run --rm train python src/monitoring/drift_simulation.py
