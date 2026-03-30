# tests/test_preprocessing.py
import pytest
import pandas as pd
from src.preprocessing.preprocessor import Preprocessor

def test_preprocessor_no_na_after():
    df = pd.DataFrame({"V1": [1, None, 3], "Class": [0, 0, 1]})
    proc = Preprocessor(scale=False)
    result = proc.fit_transform(df)
    assert result.isnull().sum().sum() == 0