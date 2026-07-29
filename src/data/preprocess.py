"""
Módulo de pré-processamento.
Interpola missings, agrega para hora e cria features temporais.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_and_aggregate(df: pd.DataFrame, target: str = "Global_active_power") -> pd.DataFrame:
    """Interpola gaps pequenos e agrega para frequência horária."""
    df = df.interpolate(method="linear", limit=60, limit_direction="both")
    df_hourly = df.resample("h").mean()

    missing_ratio = df.isnull().resample("h").mean()
    df_hourly = df_hourly[missing_ratio[target] <= 0.5]

    return df_hourly


def create_features(df: pd.DataFrame, target: str = "Global_active_power", lags: list[int] = None) -> pd.DataFrame:
    """Cria features temporais e lag features."""
    df = df.copy()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    def get_season(m: int) -> str:
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "spring"
        elif m in [6, 7, 8]:
            return "summer"
        return "autumn"

    df["season"] = df["month"].apply(get_season)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if lags:
        for lag in lags:
            df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    return df


def temporal_split(df: pd.DataFrame, test_size: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o DataFrame em treino e teste respeitando a ordem temporal."""
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
