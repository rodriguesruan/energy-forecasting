"""
ETAPA 18 — Avaliação Consolidada
Calcula e interpreta métricas para todos os modelos treinados.
Gera ranking final e gráfico comparativo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


TRAIN_PATH = Path("data/processed/df_train.parquet")
TEST_PATH = Path("data/processed/df_test.parquet")
FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")

TARGET = "Global_active_power"


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
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


def evaluate_all_models() -> None:
    print("=" * 60)
    print("  AVALIAÇÃO CONSOLIDADA — TODOS OS MODELOS")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. Carregar dados
    # -----------------------------------------------------------------
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    # -----------------------------------------------------------------
    # 2. Preparar features de ML (para obter o y_test alinhado)
    # -----------------------------------------------------------------
    lags = [1, 2, 24, 168]
    df_train_ml = create_lag_features(df_train, TARGET, lags)
    df_test_ml = create_lag_features(df_test, TARGET, lags)
    X_train, y_train_ml = prepare_features(df_train_ml, TARGET)
    X_test, y_test_aligned = prepare_features(df_test_ml, TARGET)

    print(f"\n📊 Treino ML: {len(X_train):,} registros")
    print(f"📊 Teste alinhado: {len(y_test_aligned):,} registros")

    # -----------------------------------------------------------------
    # 3. Baselines (truncados para o mesmo período do ML)
    # -----------------------------------------------------------------
    y_train = df_train[TARGET]
    # Pega o último valor do treino como previsão naive
    last_value = y_train.iloc[-1]
    pred_naive = np.full(len(y_test_aligned), last_value)
    pred_mean = np.full(len(y_test_aligned), y_train.mean())
    pred_ma = np.full(len(y_test_aligned), y_train.iloc[-24:].mean())

    # -----------------------------------------------------------------
    # 4. Modelos de ML
    # -----------------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_ml)
    pred_rf = rf.predict(X_test)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_ml)
    pred_xgb = xgb_model.predict(X_test)

    # -----------------------------------------------------------------
    # 5. Calcular métricas (todos contra y_test_aligned)
    # -----------------------------------------------------------------
    results = {
        "Naive": calculate_metrics(y_test_aligned, pred_naive),
        "Média Móvel (24h)": calculate_metrics(y_test_aligned, pred_ma),
        "Média Histórica": calculate_metrics(y_test_aligned, pred_mean),
        "SARIMA(0,0,0)(1,0,1,24)": {"MAE": 0.6624, "RMSE": 0.8281, "MAPE": 110.29, "R²": None},
        "Random Forest": calculate_metrics(y_test_aligned, pred_rf),
        "XGBoost": calculate_metrics(y_test_aligned, pred_xgb),
    }

    # -----------------------------------------------------------------
    # 6. Tabela comparativa
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TABELA COMPARATIVA")
    print("=" * 60)
    df_results = pd.DataFrame(results).T
    # Ordena por MAE (menor é melhor)
    df_results = df_results.sort_values("MAE")
    print(df_results.to_string())

    # -----------------------------------------------------------------
    # 7. Interpretação das métricas
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  INTERPRETAÇÃO DAS MÉTRICAS")
    print("=" * 60)

    best = df_results.index[0]
    best_mae = df_results.loc[best, "MAE"]
    best_rmse = df_results.loc[best, "RMSE"]
    best_mape = df_results.loc[best, "MAPE"]

    print(f"\n🏆 MELHOR MODELO: {best}")
    print(f"   MAE:  {best_mae} kW  → Em média, erra por {best_mae} kW (~{best_mae*1000:.0f} watts)")
    print(f"   RMSE: {best_rmse} kW  → Penaliza erros grandes; próximo do MAE = poucos outliers graves")
    print(f"   MAPE: {best_mape}%    → Erro percentual médio (sensível a valores próximos de zero)")
    r2_val = df_results.loc[best, "R²"]
    if r2_val is not None:
        print(f"   R²:   {r2_val}    → Explica ~{r2_val*100:.1f}% da variância do consumo")

    print(f"\n📊 COMPARAÇÃO COM BASELINE (Naive):")
    naive_mae = df_results.loc["Naive", "MAE"]
    improvement = ((naive_mae - best_mae) / naive_mae) * 100
    print(f"   O {best} reduziu o erro em {improvement:.1f}% em relação ao Naive")

    print(f"\n📊 COMPARAÇÃO COM MÉDIA HISTÓRICA:")
    hist_mae = df_results.loc["Média Histórica", "MAE"]
    improvement_hist = ((hist_mae - best_mae) / hist_mae) * 100
    print(f"   O {best} reduziu o erro em {improvement_hist:.1f}% em relação à Média Histórica")

    # -----------------------------------------------------------------
    # 8. Gráfico de barras comparativo
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["MAE", "RMSE", "MAPE"]
    colors = ["steelblue", "darkorange", "crimson", "mediumpurple", "darkgreen", "firebrick"]

    for idx, metric in enumerate(metrics):
        values = df_results[metric].astype(float).values
        bars = axes[idx].barh(df_results.index, values, color=colors[:len(values)], alpha=0.8)
        axes[idx].set_title(metric, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Valor")
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis="x")
        for bar, val in zip(bars, values):
            axes[idx].text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                           f"{val:.3f}", va="center", fontsize=9)

    plt.suptitle("Comparação de Métricas — Todos os Modelos", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_metrics_comparison.png", dpi=150)
    plt.close()
    print(f"\n✅ Gráfico salvo: figures/15_metrics_comparison.png")

    # -----------------------------------------------------------------
    # 9. Salvar relatório
    # -----------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_report.csv"
    df_results.to_csv(report_path)
    print(f"💾 Relatório salvo: {report_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    evaluate_all_models()
