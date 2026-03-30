# === FICHIER : docker/Dockerfile (VERSION CORRIGÉE) ===
FROM python:3.11.9-slim AS builder

# Installation des dépendances système nécessaires (xgboost, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip + installation des packages dans l'ordre optimal
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11.9-slim
WORKDIR /app

# Copie seulement les packages installés
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

CMD ["python", "src/train.py"]