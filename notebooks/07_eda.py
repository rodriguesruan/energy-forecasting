"""
ETAPA 12 — Análise Exploratória de Dados (EDA)
Gera gráficos profissionais para entender padrões temporais,
sazonalidade, distribuição e correlações.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path


FEATURES_PATH = Path("data/processed/df_features.parquet")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Global_active_power"


def run_eda(path: Path = FEATURES_PATH) -> None:
    df = pd.read_parquet(path)

    print("=" * 60)
    print("  ANÁLISE EXPLORATÓRIA (EDA)")
    print("=" * 60)

    # =====================================================================
    # 1. SÉRIE TEMPORAL COMPLETA
    # =====================================================================
    # Mostra a evolução do consumo ao longo dos ~4 anos.
    # Utilidade: identificar tendência de longo prazo e gaps.
    fig, ax = plt.subplots(figsize=(14, 4))
    df[TARGET].plot(ax=ax, color="steelblue", alpha=0.8)
    ax.set_title("Série Temporal Completa — Consumo de Energia (kW)", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Global Active Power (kW)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_time_series_full.png", dpi=150)
    plt.close()
    print("✅ 01_time_series_full.png salvo")

    # =====================================================================
    # 2. ZOOM EM 2 SEMANAS (para visualizar padrão diário)
    # =====================================================================
    # Utilidade: ver claramente o ciclo dia/noite e diferença entre
    # dias úteis e fins de semana.
    two_weeks = df.loc["2007-01-01":"2007-01-14"]
    fig, ax = plt.subplots(figsize=(14, 4))
    two_weeks[TARGET].plot(ax=ax, color="darkgreen", linewidth=1.2)
    ax.set_title("Zoom: 2 Semanas de Consumo (Jan/2007)", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Global Active Power (kW)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_time_series_zoom.png", dpi=150)
    plt.close()
    print("✅ 02_time_series_zoom.png salvo")

    # =====================================================================
    # 3. DECOMPOSIÇÃO SAZONAL (período = 24h)
    # =====================================================================
    # Decompõe a série em:
    #   - Tendência: comportamento de longo prazo
    #   - Sazonalidade: padrão que se repete a cada 24h
    #   - Resíduo: ruído aleatório após remover tendência e sazonalidade
    # Utilidade: confirmar se existe padrão diário forte.
    print("\n🔧 Decompondo série (período=24h)...")
    decomposition = seasonal_decompose(
        df[TARGET].dropna(),
        model="additive",
        period=24,
        extrapolate_trend="freq",
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], color="steelblue")
    axes[0].set_title("Observado")
    decomposition.trend.plot(ax=axes[1], color="darkorange")
    axes[1].set_title("Tendência")
    decomposition.seasonal.plot(ax=axes[2], color="green")
    axes[2].set_title("Sazonalidade (24h)")
    decomposition.resid.plot(ax=axes[3], color="crimson")
    axes[3].set_title("Resíduo")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_seasonal_decompose.png", dpi=150)
    plt.close()
    print("✅ 03_seasonal_decompose.png salvo")

    # =====================================================================
    # 4. HISTOGRAMA + KDE
    # =====================================================================
    # Mostra a distribuição do consumo.
    # Utilidade: identificar assimetria, múltiplos picos (multimodalidade)
    # e a presença de cauda longa (picos raros de alto consumo).
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[TARGET].dropna(), bins=80, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribuição do Consumo de Energia", fontsize=12)
    ax.set_xlabel("Global Active Power (kW)")
    ax.set_ylabel("Frequência")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_histogram.png", dpi=150)
    plt.close()
    print("✅ 04_histogram.png salvo")

    # =====================================================================
    # 5. BOXPLOT POR HORA DO DIA
    # =====================================================================
    # Compara a distribuição do consumo em cada hora.
    # Utilidade: identificar horários de pico (ex: 19h-21h jantar) e
    # horários de vale (ex: 2h-5h madrugada).
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(x="hour", y=TARGET, data=df, color="lightblue", ax=ax)
    ax.set_title("Consumo por Hora do Dia", fontsize=12)
    ax.set_xlabel("Hora")
    ax.set_ylabel("Global Active Power (kW)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_boxplot_hour.png", dpi=150)
    plt.close()
    print("✅ 05_boxplot_hour.png salvo")

    # =====================================================================
    # 6. BOXPLOT POR DIA DA SEMANA
    # =====================================================================
    # Compara consumo entre dias úteis e fins de semana.
    # Utilidade: confirmar se há diferença significativa entre
    # segunda-feira e domingo.
    day_labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="dayofweek", y=TARGET, data=df, palette="Set2", ax=ax)
    ax.set_xticklabels(day_labels)
    ax.set_title("Consumo por Dia da Semana", fontsize=12)
    ax.set_xlabel("Dia da Semana")
    ax.set_ylabel("Global Active Power (kW)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_boxplot_weekday.png", dpi=150)
    plt.close()
    print("✅ 06_boxplot_weekday.png salvo")

    # =====================================================================
    # 7. AUTOCORRELAÇÃO (ACF)
    # =====================================================================
    # Mede a correlação da série com ela mesma em diferentes lags.
    # Utilidade: identificar até quantas horas no passado ainda influenciam
    # o valor atual. Lags significativos (fora da banda azul) são úteis
    # como features para modelos de ML.
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(df[TARGET].dropna(), lags=72, ax=ax, color="steelblue")
    ax.set_title("Função de Autocorrelação (ACF) — 72h", fontsize=12)
    ax.set_xlabel("Lag (horas)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_acf.png", dpi=150)
    plt.close()
    print("✅ 07_acf.png salvo")

    # =====================================================================
    # 8. AUTOCORRELAÇÃO PARCIAL (PACF)
    # =====================================================================
    # Mede a correlação direta entre o valor atual e um lag específico,
    # REMOVENDO o efeito dos lags intermediários.
    # Utilidade: ajuda a definir o parâmetro 'p' do modelo ARIMA.
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_pacf(df[TARGET].dropna(), lags=72, ax=ax, color="darkgreen", method="ywm")
    ax.set_title("Função de Autocorrelação Parcial (PACF) — 72h", fontsize=12)
    ax.set_xlabel("Lag (horas)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_pacf.png", dpi=150)
    plt.close()
    print("✅ 08_pacf.png salvo")

    # =====================================================================
    # 9. MAPA DE CALOR DE CORRELAÇÃO
    # =====================================================================
    # Mostra correlações entre todas as variáveis numéricas.
    # Utilidade: identificar multicolinearidade e features redundantes
    # antes de treinar modelos de ML.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # máscara triangular
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Mapa de Correlação — Variáveis Numéricas", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_correlation_heatmap.png", dpi=150)
    plt.close()
    print("✅ 09_correlation_heatmap.png salvo")

    print("\n" + "=" * 60)
    print(f"  Todos os gráficos salvos em: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
