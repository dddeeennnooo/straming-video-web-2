from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def build_model():
    base = LogisticRegression(max_iter=200)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)
