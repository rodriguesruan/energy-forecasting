"""
ETAPA 20 — Intervalo de Confiança da Previsão
Gera previsões com intervalos usando Quantile Regression (XGBoost)
e método bootstrap de resampling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


TRAIN_PATH = Path("data/processed/df_train.parquet")
TEST_PATH = Path("data/processed/df_test.parquet")
FIGURES_DIR = Path("figures")

TARGET = "Global_active_power"


def create_lag_features(df: pd.DataFrame, target: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
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


def run_confidence_intervals() -> None:
    print("=" * 60)
    print("  INTERVALO DE CONFIANÇA DA PREVISÃO")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. Carregar e preparar dados
    # -----------------------------------------------------------------
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    lags = [1, 2, 24, 168]
    df_train_ml = create_lag_features(df_train, TARGET, lags)
    df_test_ml = create_lag_features(df_test, TARGET, lags)
    X_train, y_train = prepare_features(df_train_ml, TARGET)
    X_test, y_test = prepare_features(df_test_ml, TARGET)

    print(f"\n📊 Treino: {len(X_train):,} registros")
    print(f"📊 Teste:  {len(X_test):,} registros")

    # =================================================================
    # MÉTODO 1: Quantile Regression (XGBoost)
    # =================================================================
    # Regressão quantílica estima percentis diretamente.
    # Treinamos 3 modelos: P10 (limite inferior), P50 (mediana), P90 (limite superior)
    # Isso nos dá um intervalo de 80% de confiança.
    print("\n🔮 MÉTODO 1: Quantile Regression (XGBoost)")
    print("   Treinando modelo P10 (percentil 10)...")

    model_p10 = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        objective="reg:quantileerror", quantile_alpha=0.10,
    )
    model_p10.fit(X_train, y_train)

    print("   Treinando modelo P50 (mediana)...")
    model_p50 = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        objective="reg:quantileerror", quantile_alpha=0.50,
    )
    model_p50.fit(X_train, y_train)

    print("   Treinando modelo P90 (percentil 90)...")
    model_p90 = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=43, n_jobs=-1,
        objective="reg:quantileerror", quantile_alpha=0.90,
    )
    model_p90.fit(X_train, y_train)

    # Previsões
    pred_p10 = model_p10.predict(X_test)
    pred_p50 = model_p50.predict(X_test)
    pred_p90 = model_p90.predict(X_test)

    # Garante que P10 <= P50 <= P90
    pred_p10 = np.minimum(pred_p10, pred_p50)
    pred_p90 = np.maximum(pred_p90, pred_p50)

    print(f"   ✅ Intervalo 80% gerado: P10={pred_p10.mean():.3f}, P50={pred_p50.mean():.3f}, P90={pred_p90.mean():.3f}")

    # =================================================================
    # MÉTODO 2: Bootstrap (resampling dos resíduos)
    # =================================================================
    # Ideia: os erros do modelo no treino representam a incerteza.
    # Amostramos esses erros com reposição e adicionamos à previsão.
    print("\n🔮 MÉTODO 2: Bootstrap dos Resíduos")
    
    # Modelo base (mesmo da ETAPA 17)
    model_base = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model_base.fit(X_train, y_train)
    pred_base = model_base.predict(X_test)

    # Resíduos no treino
    train_pred = model_base.predict(X_train)
    train_residuals = y_train.values - train_pred

    # Bootstrap: 1000 amostras dos resíduos para cada ponto do teste
    n_bootstrap = 1000
    bootstrap_predictions = np.zeros((len(X_test), n_bootstrap))
    np.random.seed(42)
    for i in range(n_bootstrap):
        sampled_residuals = np.random.choice(train_residuals, size=len(X_test), replace=True)
        bootstrap_predictions[:, i] = pred_base + sampled_residuals

    # Percentis do bootstrap
    boot_p05 = np.percentile(bootstrap_predictions, 5, axis=1)
    boot_p50 = np.percentile(bootstrap_predictions, 50, axis=1)
    boot_p95 = np.percentile(bootstrap_predictions, 95, axis=1)

    print(f"   ✅ Intervalo 90% gerado via bootstrap")

    # =================================================================
    # 3. Gráfico comparativo (primeira semana do teste)
    # =================================================================
    n_plot = 168  # 1 semana

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- Quantile Regression ---
    axes[0].fill_between(y_test.index[:n_plot], pred_p10[:n_plot], pred_p90[:n_plot],
                         color="crimson", alpha=0.2, label="Intervalo 80% (P10-P90)")
    axes[0].plot(y_test.index[:n_plot], pred_p50[:n_plot],
                 color="crimson", linewidth=1.5, label="Mediana (P50)")
    axes[0].plot(y_test.index[:n_plot], y_test.values[:n_plot],
                 color="black", linewidth=1.5, alpha=0.8, label="Real")
    axes[0].set_title("Intervalo de Confiança — Quantile Regression (XGBoost)", fontsize=12)
    axes[0].set_ylabel("Global Active Power (kW)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # --- Bootstrap ---
    axes[1].fill_between(y_test.index[:n_plot], boot_p05[:n_plot], boot_p95[:n_plot],
                         color="steelblue", alpha=0.2, label="Intervalo 90% (P5-P95)")
    axes[1].plot(y_test.index[:n_plot], boot_p50[:n_plot],
                 color="steelblue", linewidth=1.5, label="Mediana Bootstrap")
    axes[1].plot(y_test.index[:n_plot], y_test.values[:n_plot],
                 color="black", linewidth=1.5, alpha=0.8, label="Real")
    axes[1].set_title("Intervalo de Confiança — Bootstrap dos Resíduos", fontsize=12)
    axes[1].set_xlabel("Data")
    axes[1].set_ylabel("Global Active Power (kW)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_confidence_intervals.png", dpi=150)
    plt.close()
    print(f"\n✅ Gráfico salvo: figures/21_confidence_intervals.png")

    # =================================================================
    # 4. Cobertura do intervalo (quantos valores reais caem dentro?)
    # =================================================================
    print("\n📊 COBERTURA DO INTERVALO (valores reais dentro da faixa):")
    
    # Quantile Regression: P10 a P90 = 80%
    coverage_qr = np.mean((y_test.values >= pred_p10) & (y_test.values <= pred_p90)) * 100
    print(f"   Quantile Regression (P10-P90): {coverage_qr:.1f}% (esperado: ~80%)")

    # Bootstrap: P5 a P95 = 90%
    coverage_boot = np.mean((y_test.values >= boot_p05) & (y_test.values <= boot_p95)) * 100
    print(f"   Bootstrap (P5-P95):           {coverage_boot:.1f}% (esperado: ~90%)")

    # =================================================================
    # 5. Largura média do intervalo
    # =================================================================
    width_qr = np.mean(pred_p90 - pred_p10)
    width_boot = np.mean(boot_p95 - boot_p05)
    print(f"\n📊 LARGURA MÉDIA DO INTERVALO:")
    print(f"   Quantile Regression: {width_qr:.4f} kW")
    print(f"   Bootstrap:           {width_boot:.4f} kW")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_confidence_intervals()
