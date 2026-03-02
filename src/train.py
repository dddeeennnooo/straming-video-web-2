import joblib
from pathlib import Path
from sklearn.metrics import brier_score_loss
from src.model import build_model

MODEL_PATH = Path("models/model.joblib")

def train(df):
    split = int(len(df) * 0.8)

    X_train = df.iloc[:split][["strength_diff", "form_diff", "home"]]
    y_train = df.iloc[:split]["y"]

    X_test = df.iloc[split:][["strength_diff", "form_diff", "home"]]
    y_test = df.iloc[split:]["y"]

    model = build_model()
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    print("Brier score:", brier_score_loss(y_test, p))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model
