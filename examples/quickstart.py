"""Small API demonstration; not a reproduction of the published experiments."""

from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models_hefs import HierarchicalLightGBMRegressor


def main():
    rng = np.random.default_rng(42)
    time = np.arange(800, dtype=float)
    dominant = np.column_stack([
        np.sin(time / 20), np.cos(time / 37), time / 800,
        rng.normal(size=time.size),
    ])
    complementary = rng.normal(size=(time.size, 4))
    base = 4 + dominant[:, 0] + 0.5 * dominant[:, 1]
    y = base * (1 + 0.08 * np.tanh(complementary[:, 0]))
    y += rng.normal(scale=0.08, size=time.size)
    X = np.column_stack([dominant, complementary])
    cutoff = 600
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]
    params = {"n_estimators": 100, "num_leaves": 15, "verbosity": -1, "n_jobs": 1}
    model = HierarchicalLightGBMRegressor(
        feat_split=(4, 4), beta=0.1, random_state=42,
        scale_y=False,
        lgb_params_layer1=params.copy(), lgb_params_layer2=params.copy(),
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    if prediction.shape != y_test.shape or not np.isfinite(prediction).all():
        raise RuntimeError("HEFS returned invalid predictions")
    reference = LGBMRegressor(random_state=42, **params).fit(X_train, y_train)
    print("Generated-data API demonstration (not paper results)")
    print(f"Train: {len(y_train)} samples | Test: {len(y_test)} samples | Groups: 4 + 4")
    print(f"HEFS MAE:     {mean_absolute_error(y_test, prediction):.4f}")
    print(f"LightGBM MAE: {mean_absolute_error(y_test, reference.predict(X_test)):.4f}")


if __name__ == "__main__":
    main()

