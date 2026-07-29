"""
ETAPA 7 — Carregamento dos Dados
Lê o dataset raw, aplica parse de datas, trata separador decimal
e salva versão processada para uso nas próximas etapas.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("data/raw/household_power_consumption.txt")
PROCESSED_PATH = Path("data/processed/df_raw.parquet")


def load_dataset(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Carrega o dataset de consumo de energia.

    Parâmetros:
        raw_path: caminho para o arquivo .txt original

    Retorna:
        DataFrame com datetime index e colunas numéricas
    """
    print(f"Carregando dados de: {raw_path}")

    # O arquivo usa ';' como separador e ',' como decimal.
    # Valores ausentes estão marcados com '?'.
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )

    # Combina Date + Time em uma única coluna datetime
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    # Remove colunas originais de data/hora (agora redundantes)
    df = df.drop(columns=["Date", "Time"])

    # Define datetime como índice — essencial para séries temporais
    df = df.set_index("datetime").sort_index()

    # Converte separador decimal de vírgula para ponto em todas as colunas
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"Período: {df.index.min()} → {df.index.max()}")

    return df


def save_processed(df: pd.DataFrame, output_path: Path = PROCESSED_PATH) -> None:
    """Persiste o DataFrame processado em formato Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Dados processados salvos em: {output_path}")


if __name__ == "__main__":
    df = load_dataset()
    save_processed(df)
    print("\nPrimeiras linhas:")
    print(df.head())
