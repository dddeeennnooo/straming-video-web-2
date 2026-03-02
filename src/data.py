import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def make_dataset(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    dates = [
        datetime.now() - timedelta(days=i)
        for i in range(n)
    ][::-1]

    strength_diff = rng.normal(0, 1, n)
    form_diff = rng.normal(0, 1, n)
    home = rng.integers(0, 2, n)

    logit = 0.8 * strength_diff + 0.5 * form_diff + 0.3 * home
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    return pd.DataFrame({
        "date": dates,
        "strength_diff": strength_diff,
        "form_diff": form_diff,
        "home": home,
        "y": y
    })
