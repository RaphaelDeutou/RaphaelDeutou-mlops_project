# tests/test_model.py
import pytest
from src.models.trainer import ModelTrainer
import pandas as pd
import numpy as np

def test_trainer_returns_model():
    X = pd.DataFrame(np.random.rand(100, 30))
    y = np.random.randint(0, 2, 100)
    trainer = ModelTrainer("logistic_regression")
    model = trainer.train(X, y, {"C": 1.0})
    assert hasattr(model, "predict")