"""
Módulo de carga de dados.
Lê o dataset raw e retorna um DataFrame limpo com datetime index.
"""

import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw/household_power_consumption.txt")


def load_dataset(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Carrega e faz parse inicial do dataset."""
    df = pd.read_csv(raw_path, sep=";", na_values=["?", ""], low_memory=False)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime").sort_index()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
