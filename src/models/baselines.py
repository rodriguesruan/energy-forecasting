"""
Modelos baseline para séries temporais.
"""

import numpy as np
import pandas as pd


def naive_forecast(y_train: pd.Series, n_steps: int) -> np.ndarray:
    """Previsão naive: repete o último valor do treino."""
    return np.full(n_steps, y_train.iloc[-1])


def moving_average_forecast(y_train: pd.Series, n_steps: int, window: int = 24) -> np.ndarray:
    """Previsão por média móvel dos últimos N valores do treino."""
    return np.full(n_steps, y_train.iloc[-window:].mean())


def historical_mean_forecast(y_train: pd.Series, n_steps: int) -> np.ndarray:
    """Previsão pela média histórica do treino."""
    return np.full(n_steps, y_train.mean())
