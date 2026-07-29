"""
ETAPA 11 — Engenharia de Atributos
Extrai features temporais do índice datetime para capturar padrões
de consumo: diários, semanais e sazonais.
"""

import pandas as pd
import numpy as np
from pathlib import Path


CLEAN_PATH = Path("data/processed/df_clean.parquet")
FEATURES_PATH = Path("data/processed/df_features.parquet")


def create_features(input_path: Path = CLEAN_PATH, output_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """
    Cria features temporais a partir do datetime index.
    """
    print("=" * 60)
    print("  ENGENHARIA DE ATRIBUTOS")
    print("=" * 60)

    df = pd.read_parquet(input_path).copy()
    print(f"\n📥 Dataset de entrada: {len(df):,} registros horários")

    # =========================================================================
    # FEATURES TEMPORAIS BÁSICAS
    # =========================================================================
    print("\n🔧 Criando features temporais...")

    # 1. HORA DO DIA (0-23)
    # Utilidade: captura padrão diário de consumo (pico manhã, tarde, noite)
    df["hour"] = df.index.hour

    # 2. DIA DO MÊS (1-31)
    # Utilidade: identifica padrões de início/fim de mês (contas, feriados)
    df["day"] = df.index.day

    # 3. MÊS (1-12)
    # Utilidade: sazonalidade anual — inverno consome mais energia (aquecimento)
    df["month"] = df.index.month

    # 4. ANO
    # Utilidade: detecta tendência de longo prazo (crescimento ou redução)
    df["year"] = df.index.year

    # 5. DIA DA SEMANA (0=segunda, 6=domingo)
    # Utilidade: padrão semanal — dias úteis vs. fins de semana
    df["dayofweek"] = df.index.dayofweek

    # 6. FIM DE SEMANA (booleano)
    # Utilidade: simplifica para modelos — comportamento muda drasticamente
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 7. TRIMESTRE (1-4)
    # Utilidade: agregação para análise de negócio e sazonalidade de curto prazo
    df["quarter"] = df.index.quarter

    # 8. ESTAÇÃO DO ANO (hemisfério norte — dataset da França)
    # Utilidade: inverno e verão têm picos por aquecimento/ar-condicionado
    # Inverno: dez, jan, fev | Primavera: mar, abr, mai
    # Verão: jun, jul, ago   | Outono: set, out, nov
    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    df["season"] = df["month"].apply(get_season)

    # =========================================================================
    # FEATURES CÍCLICAS (seno/cosseno)
    # =========================================================================
    # Modelos de ML tratam 23h e 0h como distantes (23 vs 0), mas na realidade
    # são adjacentes. Features cíclicas corrigem isso.
    print("🔧 Criando features cíclicas...")

    # Hora cíclica: 24h = ciclo completo
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Dia da semana cíclico: 7 dias = ciclo completo
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Mês cíclico: 12 meses = ciclo completo
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # =========================================================================
    # RESUMO
    # =========================================================================
    print(f"\n✅ Dataset enriquecido: {len(df):,} registros × {df.shape[1]} colunas")
    print("\n📋 NOVAS FEATURES CRIADAS:")
    new_cols = [c for c in df.columns if c not in [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]]
    for col in new_cols:
        print(f"  • {col}")

    # =========================================================================
    # SALVAR
    # =========================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"\n💾 Dataset com features salvo em: {output_path}")

    return df


if __name__ == "__main__":
    df = create_features()
    print("\nPrimeiras linhas com as novas features:")
    print(df.head())
