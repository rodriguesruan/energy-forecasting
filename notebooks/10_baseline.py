"""
ETAPA 15 — Modelos Baseline
Implementa modelos simples para estabelecer a linha de base
de performance. Todo modelo complexo deve superar esses números.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


TRAIN_PATH = Path("data/processed/df_train.parquet")
TEST_PATH = Path("data/processed/df_test.parquet")
FIGURES_DIR = Path("figures")

TARGET = "Global_active_power"


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de erro padrão para séries temporais.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
    }


def naive_forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """
    Naive Forecast: a previsão é o último valor observado.
    Útil quando o valor de hoje é muito parecido com o de ontem.
    """
    last_value = train.iloc[-1]
    return np.full(shape=len(test), fill_value=last_value)


def moving_average_forecast(train: pd.Series, test: pd.Series, window: int = 24) -> np.ndarray:
    """
    Média Móvel: previsão é a média dos últimos N valores do treino.
    Útil para suavizar ruído e capturar o nível recente.
    """
    # Para o primeiro ponto do teste, usamos os últimos N valores do treino
    # Para os demais, usamos os últimos N valores já previstos/observados
    history = list(train.values)
    predictions = []

    for _ in range(len(test)):
        # Pega os últimos 'window' valores disponíveis
        window_values = history[-window:]
        pred = np.mean(window_values)
        predictions.append(pred)
        # Adiciona o valor real ao history (simulação de walk-forward)
        # Na prática, em baseline simples, usamos apenas o treino
        # Aqui usamos uma abordagem mais realista: recursiva
        if len(history) < len(train) + len(test):
            # Para baseline, simplificamos: usamos média dos últimos N do treino
            pass

    # Simplificação: média móvel dos últimos N do treino para todo o teste
    last_window = train.iloc[-window:].mean()
    return np.full(shape=len(test), fill_value=last_window)


def historical_mean_forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """
    Média Histórica: previsão é a média de todo o conjunto de treino.
    Útil como baseline mínimo — qualquer modelo deve superar isso.
    """
    mean_value = train.mean()
    return np.full(shape=len(test), fill_value=mean_value)


def run_baselines() -> None:
    print("=" * 60)
    print("  MODELOS BASELINE")
    print("=" * 60)

    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    y_train = df_train[TARGET]
    y_test = df_test[TARGET]

    print(f"\n📊 Treino: {len(y_train):,} registros")
    print(f"📊 Teste:  {len(y_test):,} registros")

    # -----------------------------------------------------------------
    # 1. Naive Forecast
    # -----------------------------------------------------------------
    print("\n🔮 NAIVE FORECAST (último valor do treino)")
    pred_naive = naive_forecast(y_train, y_test)
    metrics_naive = calculate_metrics(y_test, pred_naive)
    print(f"   MAE:  {metrics_naive['MAE']:.4f} kW")
    print(f"   RMSE: {metrics_naive['RMSE']:.4f} kW")
    print(f"   MAPE: {metrics_naive['MAPE']:.2f}%")

    # -----------------------------------------------------------------
    # 2. Média Móvel (24h)
    # -----------------------------------------------------------------
    print("\n📈 MÉDIA MÓVEL (24h)")
    pred_ma = moving_average_forecast(y_train, y_test, window=24)
    metrics_ma = calculate_metrics(y_test, pred_ma)
    print(f"   MAE:  {metrics_ma['MAE']:.4f} kW")
    print(f"   RMSE: {metrics_ma['RMSE']:.4f} kW")
    print(f"   MAPE: {metrics_ma['MAPE']:.2f}%")

    # -----------------------------------------------------------------
    # 3. Média Histórica
    # -----------------------------------------------------------------
    print("\n📊 MÉDIA HISTÓRICA")
    pred_mean = historical_mean_forecast(y_train, y_test)
    metrics_mean = calculate_metrics(y_test, pred_mean)
    print(f"   MAE:  {metrics_mean['MAE']:.4f} kW")
    print(f"   RMSE: {metrics_mean['RMSE']:.4f} kW")
    print(f"   MAPE: {metrics_mean['MAPE']:.2f}%")

    # -----------------------------------------------------------------
    # 4. Resumo comparativo
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESUMO DOS BASELINES")
    print("=" * 60)
    summary = pd.DataFrame({
        "Naive": metrics_naive,
        "Média Móvel (24h)": metrics_ma,
        "Média Histórica": metrics_mean,
    })
    print(summary.T.to_string())

    # -----------------------------------------------------------------
    # 5. Gráfico comparativo (primeiras 168h = 1 semana do teste)
    # -----------------------------------------------------------------
    n_plot = 168  # 1 semana
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(y_test.index[:n_plot], y_test.values[:n_plot], 
            label="Real", color="black", linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index[:n_plot], pred_naive[:n_plot], 
            label="Naive", color="crimson", linestyle="--", alpha=0.7)
    ax.plot(y_test.index[:n_plot], pred_ma[:n_plot], 
            label="Média Móvel 24h", color="darkorange", linestyle="--", alpha=0.7)
    ax.plot(y_test.index[:n_plot], pred_mean[:n_plot], 
            label="Média Histórica", color="steelblue", linestyle="--", alpha=0.7)

    ax.set_title("Baseline vs. Real — Primeira Semana do Teste", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_baseline_comparison.png", dpi=150)
    plt.close()
    print(f"\n✅ Gráfico salvo: figures/11_baseline_comparison.png")

    print("\n" + "=" * 60)
    print("  Próximos modelos devem superar o melhor baseline acima.")
    print("=" * 60)


if __name__ == "__main__":
    run_baselines()
