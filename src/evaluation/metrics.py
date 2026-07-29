"""
Módulo de métricas de avaliação.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calcula MAE, RMSE, MAPE e R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "R²": round(r2, 4),
    }
