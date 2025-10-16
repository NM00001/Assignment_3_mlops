from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import root_mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def to_metrics_dict(rmse_value: float) -> Dict[str, float]:
    return {"rmse": round(rmse_value, 4)}
