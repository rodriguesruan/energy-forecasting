"""
ETAPA 19 — Visualização de Diagnóstico
Gera gráficos profissionais: previsão vs real, resíduos,
distribuição de erros e autocorrelação dos resíduos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
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


def run_visualization() -> None:
    print("=" * 60)
    print("  VISUALIZAÇÃO DE DIAGNÓSTICO — XGBoost")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. Carregar e preparar dados
    # -----------------------------------------------------------------
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    lags = [1, 2, 24, 168]
    df_train_ml = create_lag_features(df_train, TARGET, lags)
    df_test_ml = create_lag_features(df_test, TARGET, lags)
    X_train, y_train_ml = prepare_features(df_train_ml, TARGET)
    X_test, y_test = prepare_features(df_test_ml, TARGET)

    # -----------------------------------------------------------------
    # 2. Treinar XGBoost (melhor modelo)
    # -----------------------------------------------------------------
    print("\n🔧 Retreinando XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train_ml)
    y_pred = model.predict(X_test)

    # Resíduos
    residuals = y_test.values - y_pred

    print(f"   Previsões geradas: {len(y_pred):,}")
    print(f"   Resíduos calculados: média={residuals.mean():.4f}, std={residuals.std():.4f}")

    # -----------------------------------------------------------------
    # 3. Gráfico 1: Previsão vs Real (primeira semana do teste)
    # -----------------------------------------------------------------
    n_plot = 168  # 1 semana
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test.index[:n_plot], y_test.values[:n_plot],
            label="Real", color="black", linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index[:n_plot], y_pred[:n_plot],
            label="XGBoost — Previsão", color="crimson", linewidth=1.5, alpha=0.8)
    ax.fill_between(y_test.index[:n_plot],
                    y_pred[:n_plot] - np.abs(residuals[:n_plot]),
                    y_pred[:n_plot] + np.abs(residuals[:n_plot]),
                    color="crimson", alpha=0.1, label="Erro absoluto")
    ax.set_title("XGBoost: Previsão vs Real — Primeira Semana do Teste", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_prediction_vs_real.png", dpi=150)
    plt.close()
    print("✅ 16_prediction_vs_real.png salvo")

    # -----------------------------------------------------------------
    # 4. Gráfico 2: Resíduos ao longo do tempo
    # -----------------------------------------------------------------
    # Se houver padrão nos resíduos (ex: sempre positivo em certo período),
    # o modelo está enviesado naquele período.
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_test.index, residuals, color="steelblue", alpha=0.6, linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Resíduos ao Longo do Tempo (XGBoost)", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Erro (Real - Previsto) [kW]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_residuals_time.png", dpi=150)
    plt.close()
    print("✅ 17_residuals_time.png salvo")

    # -----------------------------------------------------------------
    # 5. Gráfico 3: Distribuição dos resíduos
    # -----------------------------------------------------------------
    # Resíduos normais (em forma de sino) indicam que o erro é aleatório.
    # Assimetria ou caudas pesadas indicam problemas no modelo.
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(residuals, bins=80, kde=True, color="steelblue", ax=ax)
    ax.axvline(x=0, color="crimson", linestyle="--", linewidth=2, label="Erro zero")
    ax.set_title("Distribuição dos Resíduos (XGBoost)", fontsize=12)
    ax.set_xlabel("Erro (Real - Previsto) [kW]")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_residuals_distribution.png", dpi=150)
    plt.close()
    print("✅ 18_residuals_distribution.png salvo")

    # -----------------------------------------------------------------
    # 6. Gráfico 4: Resíduos vs Valores Previstos (heterocedasticidade)
    # -----------------------------------------------------------------
    # Se o "funil" se abre (variância do erro aumenta com o valor previsto),
    # há heterocedasticidade — o modelo é menos confiável para valores altos.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    ax.axhline(y=0, color="crimson", linestyle="--", linewidth=1)
    ax.set_title("Resíduos vs Valores Previstos (XGBoost)", fontsize=12)
    ax.set_xlabel("Valor Previsto (kW)")
    ax.set_ylabel("Erro (Real - Previsto) [kW]")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "19_residuals_vs_predicted.png", dpi=150)
    plt.close()
    print("✅ 19_residuals_vs_predicted.png salvo")

    # -----------------------------------------------------------------
    # 7. Gráfico 5: Scatter Previsão vs Real
    # -----------------------------------------------------------------
    # Pontos próximos da linha diagonal = previsões perfeitas.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test.values, y_pred, alpha=0.3, s=10, color="steelblue")
    # Linha diagonal de referência
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2, label="Perfeito (y=x)")
    ax.set_title("Previsão vs Real — Scatter (XGBoost)", fontsize=12)
    ax.set_xlabel("Valor Real (kW)")
    ax.set_ylabel("Valor Previsto (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_scatter_real_vs_pred.png", dpi=150)
    plt.close()
    print("✅ 20_scatter_real_vs_pred.png salvo")

    # -----------------------------------------------------------------
    # 8. Estatísticas dos resíduos
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ESTATÍSTICAS DOS RESÍDUOS (XGBoost)")
    print("=" * 60)
    print(f"  Média dos resíduos:     {residuals.mean():.6f} kW (ideal: ~0)")
    print(f"  Desvio padrão:          {residuals.std():.4f} kW")
    print(f"  Mínimo:                 {residuals.min():.4f} kW")
    print(f"  Máximo:                 {residuals.max():.4f} kW")
    print(f"  Assimetria (skewness):  {pd.Series(residuals).skew():.4f} (ideal: ~0)")
    print(f"  Curtose (kurtosis):     {pd.Series(residuals).kurtosis():.4f} (ideal: ~3 para normal)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_visualization()
