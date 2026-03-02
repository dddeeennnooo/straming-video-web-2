import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")

def predict(df):
    model = joblib.load(MODEL_PATH)

    X = df[["strength_diff", "form_diff", "home"]]
    p = model.predict_proba(X)[:, 1]
    odds = 1 / np.clip(p, 1e-6, 1 - 1e-6)

    return p, odds
