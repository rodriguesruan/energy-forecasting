"""
ETAPA 16 — Modelos ARIMA e SARIMA
Implementa modelos clássicos de séries temporais e compara com baselines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


def safe_fit(model, method="lbfgs"):
    """
    Tenta fit com lbfgs; se falhar, tenta com nm (Nelder-Mead).
    NÃO passa 'disp' para evitar o bug de compatibilidade.
    """
    try:
        return model.fit(method=method)
    except Exception as e:
        print(f"   ⚠️  Falha com method='{method}': {e}")
        if method != "nm":
            print(f"   🔄 Tentando method='nm' (Nelder-Mead)...")
            return model.fit(method="nm")
        raise


def run_sarima() -> None:
    print("=" * 60)
    print("  MODELOS ARIMA / SARIMA")
    print("=" * 60)

    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    y_train = df_train[TARGET]
    y_test = df_test[TARGET]

    print(f"\n📊 Treino: {len(y_train):,} registros")
    print(f"📊 Teste:  {len(y_test):,} registros")

    # =================================================================
    # 1. ARIMA(1, 0, 1) — sem sazonalidade
    # =================================================================
    print("\n🔧 Treinando ARIMA(1, 0, 1)...")
    try:
        model_arima = SARIMAX(y_train, order=(1, 0, 1), enforce_stationarity=False)
        result_arima = safe_fit(model_arima)
        pred_arima = result_arima.forecast(steps=len(y_test))
        pred_arima = pd.Series(pred_arima.values, index=y_test.index)
        metrics_arima = calculate_metrics(y_test, pred_arima)
        print(f"   ✅ ARIMA(1,0,1) — MAE: {metrics_arima['MAE']}, RMSE: {metrics_arima['RMSE']}, MAPE: {metrics_arima['MAPE']}%")
    except Exception as e:
        print(f"   ❌ Erro no ARIMA: {e}")
        pred_arima = pd.Series(np.full(len(y_test), y_train.mean()), index=y_test.index)
        metrics_arima = {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf}

    # =================================================================
    # 2. SARIMA simplificado — sazonalidade diária
    # =================================================================
    # Usamos (0,0,0)(1,0,1,24) para ser mais leve e convergir mais rápido
    print("\n🔧 Treinando SARIMA(0,0,0)(1,0,1,24)...")
    print("   (Modelo sazonal puro — mais leve e estável)")
    try:
        model_sarima = SARIMAX(
            y_train,
            order=(0, 0, 0),
            seasonal_order=(1, 0, 1, 24),
            enforce_stationarity=False,
        )
        result_sarima = safe_fit(model_sarima)
        pred_sarima = result_sarima.forecast(steps=len(y_test))
        pred_sarima = pd.Series(pred_sarima.values, index=y_test.index)
        metrics_sarima = calculate_metrics(y_test, pred_sarima)
        print(f"   ✅ SARIMA(0,0,0)(1,0,1,24) — MAE: {metrics_sarima['MAE']}, RMSE: {metrics_sarima['RMSE']}, MAPE: {metrics_sarima['MAPE']}%")
    except Exception as e:
        print(f"   ❌ Erro no SARIMA: {e}")
        pred_sarima = pd.Series(np.full(len(y_test), y_train.mean()), index=y_test.index)
        metrics_sarima = {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf}

    # =================================================================
    # 3. Resumo comparativo
    # =================================================================
    print("\n" + "=" * 60)
    print("  RESUMO — ARIMA / SARIMA vs. BASELINES")
    print("=" * 60)
    summary = pd.DataFrame({
        "Naive": {"MAE": 0.6688, "RMSE": 0.9803, "MAPE": 47.90},
        "Média Histórica": {"MAE": 0.6350, "RMSE": 0.7498, "MAPE": 109.54},
        "ARIMA(1,0,1)": metrics_arima,
        "SARIMA(0,0,0)(1,0,1,24)": metrics_sarima,
    }).T
    print(summary.to_string())

    # =================================================================
    # 4. Gráfico comparativo (primeira semana do teste)
    # =================================================================
    n_plot = 168  # 1 semana
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(y_test.index[:n_plot], y_test.values[:n_plot],
            label="Real", color="black", linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index[:n_plot], np.full(n_plot, y_train.mean()),
            label="Média Histórica (baseline)", color="steelblue", linestyle="--", alpha=0.6)
    ax.plot(y_test.index[:n_plot], pred_arima.iloc[:n_plot],
            label="ARIMA(1,0,1)", color="darkorange", linestyle="--", alpha=0.8)
    ax.plot(y_test.index[:n_plot], pred_sarima.iloc[:n_plot],
            label="SARIMA(0,0,0)(1,0,1,24)", color="crimson", linestyle="--", alpha=0.8)

    ax.set_title("ARIMA / SARIMA vs. Real — Primeira Semana do Teste", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_arima_sarima_comparison.png", dpi=150)
    plt.close()
    print(f"\n✅ Gráfico salvo: figures/12_arima_sarima_comparison.png")

    # =================================================================
    # 5. Intervalo de confiança (SARIMA)
    # =================================================================
    print("\n🔮 Gerando previsão com intervalo de confiança (SARIMA)...")
    try:
        forecast_obj = result_sarima.get_forecast(steps=len(y_test))
        conf_int = forecast_obj.conf_int()
        pred_mean = forecast_obj.predicted_mean
        pred_mean = pd.Series(pred_mean.values, index=y_test.index)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(y_test.index[:n_plot], y_test.values[:n_plot],
                label="Real", color="black", linewidth=1.5, alpha=0.8)
        ax.plot(y_test.index[:n_plot], pred_mean.iloc[:n_plot],
                label="SARIMA — Previsão", color="crimson", linewidth=1.5)
        ax.fill_between(
            y_test.index[:n_plot],
            conf_int.iloc[:n_plot, 0],
            conf_int.iloc[:n_plot, 1],
            color="crimson", alpha=0.15, label="Intervalo 95%"
        )
        ax.set_title("SARIMA com Intervalo de Confiança 95%", fontsize=12)
        ax.set_xlabel("Data")
        ax.set_ylabel("Global Active Power (kW)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "13_sarima_confidence_interval.png", dpi=150)
        plt.close()
        print("✅ Intervalo de confiança salvo: figures/13_sarima_confidence_interval.png")
    except Exception as e:
        print(f"   ⚠️ Não foi possível gerar intervalo: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_sarima()
