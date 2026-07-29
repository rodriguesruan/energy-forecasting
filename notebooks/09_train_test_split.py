"""
ETAPA 14 — Separação Treino/Teste
Divide a série temporal respeitando a ordem cronológica.
NUNCA usar train_test_split aleatório em séries temporais.
"""

import pandas as pd
from pathlib import Path


FEATURES_PATH = Path("data/processed/df_features.parquet")
TRAIN_PATH = Path("data/processed/df_train.parquet")
TEST_PATH = Path("data/processed/df_test.parquet")

TARGET = "Global_active_power"
TEST_SIZE = 0.20  # 20% dos dados mais recentes para teste


def temporal_split(
    input_path: Path = FEATURES_PATH,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide a série temporal em treino e teste de forma cronológica.

    Parâmetros:
        test_size: proporção do dataset mais recente reservada para teste

    Retorna:
        (df_train, df_test) — ambos ordenados cronologicamente
    """
    print("=" * 60)
    print("  SEPARAÇÃO TREINO / TESTE")
    print("=" * 60)

    df = pd.read_parquet(input_path).sort_index()
    print(f"\n📥 Dataset completo: {len(df):,} registros")
    print(f"   Período: {df.index.min()} → {df.index.max()}")

    # -----------------------------------------------------------------
    # Corte temporal: os últimos 20% dos dados são o teste
    # -----------------------------------------------------------------
    split_idx = int(len(df) * (1 - test_size))
    split_date = df.index[split_idx]

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"\n✂️  Ponto de corte: {split_date}")
    print(f"   Treino: {len(df_train):,} registros ({len(df_train)/len(df)*100:.1f}%)")
    print(f"           {df_train.index.min()} → {df_train.index.max()}")
    print(f"   Teste:  {len(df_test):,} registros ({len(df_test)/len(df)*100:.1f}%)")
    print(f"           {df_test.index.min()} → {df_test.index.max()}")

    # -----------------------------------------------------------------
    # Verificações de segurança
    # -----------------------------------------------------------------
    print("\n🔒 VERIFICAÇÕES DE SEGURANÇA:")
    assert df_train.index.max() < df_test.index.min(), \
        "ERRO: treino e teste se sobrepõem!"
    print("   ✓ Nenhuma sobreposição entre treino e teste")

    assert df_train.index.is_monotonic_increasing, \
        "ERRO: treino não está ordenado!"
    print("   ✓ Treino está em ordem cronológica")

    assert df_test.index.is_monotonic_increasing, \
        "ERRO: teste não está ordenado!"
    print("   ✓ Teste está em ordem cronológica")

    # -----------------------------------------------------------------
    # Persistir
    # -----------------------------------------------------------------
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(train_path)
    df_test.to_parquet(test_path)

    print(f"\n💾 Treino salvo: {train_path}")
    print(f"💾 Teste salvo:  {test_path}")

    print("\n" + "=" * 60)

    return df_train, df_test


if __name__ == "__main__":
    df_train, df_test = temporal_split()
