"""
Modelos de Machine Learning para previsão de séries temporais.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def prepare_ml_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara X e y para modelos de ML (sem data leakage)."""
    feature_cols = [
        "hour", "dayofweek", "month", "quarter", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    lag_cols = [c for c in df.columns if "_lag_" in c]
    feature_cols.extend(lag_cols)

    X = df[feature_cols].copy()
    y = df[target].copy()
    mask = X.notnull().all(axis=1) & y.notnull()
    return X[mask], y[mask]


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Treina Random Forest."""
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """Treina XGBoost."""
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
