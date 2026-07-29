"""
ETAPA 13 — Verificação de Estacionariedade (ADF)
Aplica o teste Augmented Dickey-Fuller na série original e nas
séries transformadas (diferenciação e log).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path


FEATURES_PATH = Path("data/processed/df_features.parquet")
FIGURES_DIR = Path("figures")
TARGET = "Global_active_power"


def adf_test(series: pd.Series, title: str = "Série") -> dict:
    """
    Executa o teste ADF e retorna resultados formatados.

    Hipótese nula (H0): a série possui raiz unitária (NÃO é estacionária)
    Hipótese alternativa (H1): a série NÃO possui raiz unitária (é estacionária)

    Se p-valor < 0.05 → rejeitamos H0 → série é estacionária
    """
    result = adfuller(series.dropna(), autolag="AIC")

    print(f"\n{'='*55}")
    print(f"  TESTE ADF — {title}")
    print(f"{'='*55}")
    print(f"  Estatística ADF:        {result[0]:.6f}")
    print(f"  p-valor:                {result[1]:.6f}")
    print(f"  Valores críticos:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.6f}", end="")
        if result[0] < value:
            print("  ← estatística < crítico (rejeita H0)")
        else:
            print("  ← estatística > crítico (não rejeita H0)")

    if result[1] < 0.05:
        print(f"\n  ✅ CONCLUSÃO: p-valor < 0.05 → SÉRIE É ESTACIONÁRIA")
    else:
        print(f"\n  ❌ CONCLUSÃO: p-valor >= 0.05 → SÉRIE NÃO É ESTACIONÁRIA")

    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def plot_series_comparison(
    original: pd.Series,
    diff1: pd.Series,
    log_diff1: pd.Series,
    output_path: Path,
) -> None:
    """
    Plota a série original, a primeira diferenciação e a log-diferenciada
    lado a lado para comparação visual.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    original.plot(ax=axes[0], color="steelblue", alpha=0.8)
    axes[0].set_title("Série Original — Global Active Power (kW)", fontsize=11)
    axes[0].set_ylabel("kW")
    axes[0].grid(True, alpha=0.3)

    diff1.plot(ax=axes[1], color="darkorange", alpha=0.8)
    axes[1].set_title("1ª Diferenciação (diff)", fontsize=11)
    axes[1].set_ylabel("Δ kW")
    axes[1].grid(True, alpha=0.3)

    log_diff1.plot(ax=axes[2], color="darkgreen", alpha=0.8)
    axes[2].set_title("Log + 1ª Diferenciação (log-diff)", fontsize=11)
    axes[2].set_ylabel("Δ log(kW)")
    axes[2].set_xlabel("Data")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n✅ Gráfico salvo: {output_path}")


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    series = df[TARGET].dropna()

    print("=" * 60)
    print("  VERIFICAÇÃO DE ESTACIONARIEDADE")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. Série Original
    # -----------------------------------------------------------------
    adf_original = adf_test(series, "Série Original")

    # -----------------------------------------------------------------
    # 2. 1ª Diferenciação
    # -----------------------------------------------------------------
    # A diferenciação remove tendência: y_t' = y_t - y_{t-1}
    # Se a série original tem tendência, a diferenciada geralmente fica estacionária.
    # Isso define o parâmetro 'd' do ARIMA: d=1 significa 1 diferenciação.
    diff1 = series.diff().dropna()
    adf_diff1 = adf_test(diff1, "1ª Diferenciação")

    # -----------------------------------------------------------------
    # 3. Log + 1ª Diferenciação
    # -----------------------------------------------------------------
    # A transformação log estabiliza a variância (reduz efeito de picos).
    # Útil quando a variância cresce com o nível da série.
    log_series = np.log(series)
    log_diff1 = log_series.diff().dropna()
    adf_log_diff1 = adf_test(log_diff1, "Log + 1ª Diferenciação")

    # -----------------------------------------------------------------
    # 4. Visualização comparativa
    # -----------------------------------------------------------------
    plot_series_comparison(series, diff1, log_diff1, FIGURES_DIR / "10_stationarity_comparison.png")

    # -----------------------------------------------------------------
    # 5. Resumo
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESUMO DA ESTACIONARIEDADE")
    print("=" * 60)
    print(f"  Original:      {'ESTACIONÁRIA' if adf_original['is_stationary'] else 'NÃO ESTACIONÁRIA'}")
    print(f"  Diff(1):       {'ESTACIONÁRIA' if adf_diff1['is_stationary'] else 'NÃO ESTACIONÁRIA'}")
    print(f"  Log + Diff(1): {'ESTACIONÁRIA' if adf_log_diff1['is_stationary'] else 'NÃO ESTACIONÁRIA'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
