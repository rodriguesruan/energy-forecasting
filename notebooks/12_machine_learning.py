"""
ETAPA 17 — Modelos de Machine Learning (CORRIGIDO)
Remove data leakage ao excluir features exógenas do mesmo instante.
Usa apenas features temporais + lags do target.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


TRAIN_PATH = Path("data/processed/df_train.parquet")
TEST_PATH = Path("data/processed/df_test.parquet")
FIGURES_DIR = Path("figures")

TARGET = "Global_active_power"


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


def create_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    """Cria lag features do target (valores passados como preditores)."""
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X e y usando APENAS:
    - Features temporais (hora, dia, etc.)
    - Lags do target (valores passados)
    
    NÃO inclui features exógenas do mesmo instante (data leakage!).
    """
    feature_cols = [
        "hour", "dayofweek", "month", "quarter", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    
    # Adiciona lag features se existirem
    lag_cols = [c for c in df.columns if "_lag_" in c]
    feature_cols.extend(lag_cols)

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Remove linhas com NaN (causados pelos lags no início da série)
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    return X, y


def run_ml_models() -> None:
    print("=" * 60)
    print("  MODELOS DE MACHINE LEARNING (SEM DATA LEAKAGE)")
    print("=" * 60)

    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    print(f"\n📊 Treino: {len(df_train):,} registros")
    print(f"📊 Teste:  {len(df_test):,} registros")

    # -----------------------------------------------------------------
    # 1. Criar lag features (1h, 2h, 24h, 168h=1 semana)
    # -----------------------------------------------------------------
    print("\n🔧 Criando lag features (1h, 2h, 24h, 168h)...")
    lags = [1, 2, 24, 168]
    
    df_train = create_lag_features(df_train, TARGET, lags)
    df_test = create_lag_features(df_test, TARGET, lags)

    # -----------------------------------------------------------------
    # 2. Preparar X e y
    # -----------------------------------------------------------------
    X_train, y_train = prepare_features(df_train, TARGET)
    X_test, y_test = prepare_features(df_test, TARGET)

    print(f"   Features usadas: {list(X_train.columns)}")
    print(f"   Amostras de treino após lag: {len(X_train):,}")
    print(f"   Amostras de teste após lag:  {len(X_test):,}")

    # -----------------------------------------------------------------
    # 3. Random Forest
    # -----------------------------------------------------------------
    print("\n🌲 Treinando Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    metrics_rf = calculate_metrics(y_test, pred_rf)
    print(f"   ✅ Random Forest — MAE: {metrics_rf['MAE']}, RMSE: {metrics_rf['RMSE']}, MAPE: {metrics_rf['MAPE']}%")

    # -----------------------------------------------------------------
    # 4. XGBoost
    # -----------------------------------------------------------------
    print("\n🚀 Treinando XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)
    metrics_xgb = calculate_metrics(y_test, pred_xgb)
    print(f"   ✅ XGBoost — MAE: {metrics_xgb['MAE']}, RMSE: {metrics_xgb['RMSE']}, MAPE: {metrics_xgb['MAPE']}%")

    # -----------------------------------------------------------------
    # 5. Feature Importance (Random Forest)
    # -----------------------------------------------------------------
    print("\n📊 Feature Importance (Random Forest) — SEM DATA LEAKAGE:")
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    print(importance.to_string())

    # -----------------------------------------------------------------
    # 6. Resumo comparativo
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESUMO — TODOS OS MODELOS")
    print("=" * 60)
    summary = pd.DataFrame({
        "Naive": {"MAE": 0.6688, "RMSE": 0.9803, "MAPE": 47.90},
        "Média Histórica": {"MAE": 0.6350, "RMSE": 0.7498, "MAPE": 109.54},
        "SARIMA(0,0,0)(1,0,1,24)": {"MAE": 0.6624, "RMSE": 0.8281, "MAPE": 110.29},
        "Random Forest": metrics_rf,
        "XGBoost": metrics_xgb,
    }).T
    print(summary.to_string())

    # -----------------------------------------------------------------
    # 7. Gráfico comparativo
    # -----------------------------------------------------------------
    n_plot = 168  # 1 semana
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(y_test.index[:n_plot], y_test.values[:n_plot],
            label="Real", color="black", linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index[:n_plot], np.full(n_plot, y_test.mean()),
            label="Média Histórica", color="steelblue", linestyle="--", alpha=0.5)
    ax.plot(y_test.index[:n_plot], pred_rf[:n_plot],
            label="Random Forest", color="darkgreen", linestyle="--", alpha=0.8)
    ax.plot(y_test.index[:n_plot], pred_xgb[:n_plot],
            label="XGBoost", color="crimson", linestyle="--", alpha=0.8)

    ax.set_title("Machine Learning (Sem Data Leakage) vs. Real", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_ml_comparison.png", dpi=150)
    plt.close()
    print(f"\n✅ Gráfico salvo: figures/14_ml_comparison.png")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_ml_models()
