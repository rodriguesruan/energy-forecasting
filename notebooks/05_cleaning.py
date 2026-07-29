"""
ETAPA 10 — Limpeza dos Dados
Trata missings, agrega para granularidade horária e cria flags de qualidade.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PARQUET = Path("data/processed/df_raw.parquet")
CLEAN_PARQUET = Path("data/processed/df_clean.parquet")


def clean_dataset(raw_path: Path = RAW_PARQUET, output_path: Path = CLEAN_PARQUET) -> pd.DataFrame:
    """
    Pipeline de limpeza:
      1. Carrega dados brutos
      2. Interpola gaps pequenos (<= 60 minutos)
      3. Agrega para frequência horária (média)
      4. Remove horas com dados insuficientes (> 50% missing na hora)
      5. Salva versão limpa
    """
    print("=" * 60)
    print("  LIMPEZA DOS DADOS")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Carregar
    # -------------------------------------------------------------------------
    df = pd.read_parquet(raw_path)
    print(f"\n📥 Dataset bruto: {df.shape[0]:,} registros (minuto a minuto)")

    # -------------------------------------------------------------------------
    # 2. Interpolação de gaps pequenos (<= 60 minutos consecutivos)
    # -------------------------------------------------------------------------
    # Regra da indústria: gaps de 1 hora ou menos são seguros para interpolação
    # linear em consumo de energia (o padrão não muda drasticamente em 1h)
    print("\n🔧 Interpolando gaps pequenos (<= 60 minutos)...")
    n_before = df.isnull().sum().sum()

    # Interpolação linear para todos os gaps (limita a 60 passos consecutivos)
    df_interp = df.interpolate(method="linear", limit=60, limit_direction="both")

    n_after = df_interp.isnull().sum().sum()
    print(f"  Valores preenchidos: {n_before - n_after:,}")

    # -------------------------------------------------------------------------
    # 3. Agregação horária
    # -------------------------------------------------------------------------
    # Por que agregar?
    # - Reduz ruído de medição minuto a minuto
    # - Torna o treinamento de ARIMA/SARIMA viável (2M de pontos é excessivo)
    # - Permite identificar padrões diários/semanais mais claramente
    print("\n📊 Agregando para frequência horária (média)...")
    df_hourly = df_interp.resample("h").mean()

    # Flag: proporção de missings originais naquela hora
    # Se mais de 50% dos minutos da hora eram NaN, descartamos a hora
    missing_ratio = df.isnull().resample("h").mean()
    df_hourly = df_hourly[missing_ratio["Global_active_power"] <= 0.5]

    print(f"  Registros horários gerados: {len(df_hourly):,}")
    print(f"  Período: {df_hourly.index.min()} → {df_hourly.index.max()}")

    # -------------------------------------------------------------------------
    # 4. Verificação final
    # -------------------------------------------------------------------------
    print("\n✅ VERIFICAÇÃO FINAL")
    print(f"  Missings restantes: {df_hourly.isnull().sum().sum()}")
    print(f"  Duplicatas de timestamp: {df_hourly.index.duplicated().sum()}")
    print(f"  Valor mínimo de potência: {df_hourly['Global_active_power'].min():.3f} kW")
    print(f"  Valor máximo de potência: {df_hourly['Global_active_power'].max():.3f} kW")

    # -------------------------------------------------------------------------
    # 5. Salvar
    # -------------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_hourly.to_parquet(output_path)
    print(f"\n💾 Dataset limpo salvo em: {output_path}")

    return df_hourly


if __name__ == "__main__":
    df_clean = clean_dataset()
    print("\nPrimeiras linhas do dataset limpo:")
    print(df_clean.head())
