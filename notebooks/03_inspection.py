"""
ETAPA 8 — Inspeção Inicial dos Dados
Gera um relatório completo de saúde do dataset processado.
"""

import pandas as pd
from pathlib import Path


PROCESSED_PATH = Path("data/processed/df_raw.parquet")


def inspect_dataset(path: Path = PROCESSED_PATH) -> None:
    """
    Executa inspeção inicial completa do dataset.
    """
    df = pd.read_parquet(path)

    print("=" * 60)
    print("  INSPEÇÃO INICIAL DO DATASET")
    print("=" * 60)

    # 1. Shape
    print(f"\n📐 SHAPE: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    # 2. Info (tipos e memória)
    print("\n📋 INFO:")
    df.info()

    # 3. Describe (estatísticas descritivas)
    print("\n📊 DESCRIBE:")
    print(df.describe().T.round(3))

    # 4. Tipos de dados
    print("\n🔤 TIPOS DE DADOS:")
    print(df.dtypes)

    # 5. Valores ausentes
    print("\n❌ VALORES AUSENTES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "ausentes": missing,
        "percentual": missing_pct
    })
    print(missing_df[missing_df["ausentes"] > 0])
    if missing_df["ausentes"].sum() == 0:
        print("Nenhum valor ausente encontrado.")

    # 6. Duplicatas
    print("\n🔄 DUPLICATAS:")
    n_duplicados = df.duplicated().sum()
    print(f"Registros duplicados: {n_duplicados:,} ({n_duplicados/len(df)*100:.4f}%)")

    # 7. Resumo temporal
    print("\n⏰ RESUMO TEMPORAL:")
    print(f"  Primeiro registro: {df.index.min()}")
    print(f"  Último registro:   {df.index.max()}")
    print(f"  Frequência:        a cada minuto")
    print(f"  Total de dias:     {(df.index.max() - df.index.min()).days}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    inspect_dataset()
