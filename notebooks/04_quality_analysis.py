"""
ETAPA 9 — Análise da Qualidade dos Dados
Investiga padrões de missings, duplicatas, outliers e consistência física.
"""

import pandas as pd
import numpy as np
from pathlib import Path


PROCESSED_PATH = Path("data/processed/df_raw.parquet")


def analyze_quality(path: Path = PROCESSED_PATH) -> None:
    df = pd.read_parquet(path)

    print("=" * 60)
    print("  ANÁLISE DE QUALIDADE DOS DADOS")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Padrão dos Missings
    # -------------------------------------------------------------------------
    print("\n🔍 PADRÃO DOS VALORES AUSENTES")
    missing_mask = df["Global_active_power"].isnull()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        # Agrupa missings consecutivos em blocos
        missing_groups = (missing_mask != missing_mask.shift()).cumsum()
        block_sizes = missing_mask.groupby(missing_groups).sum()
        block_sizes = block_sizes[block_sizes > 0]

        print(f"  Total de blocos de missings: {len(block_sizes)}")
        print(f"  Maior bloco contínuo: {block_sizes.max()} minutos")
        print(f"  Menor bloco: {block_sizes.min()} minutos")
        print(f"  Média do bloco: {block_sizes.mean():.1f} minutos")
        print(f"  Distribuição dos tamanhos de bloco:")
        print(block_sizes.value_counts().head(10).sort_index())
    else:
        print("  Nenhum missing encontrado.")

    # -------------------------------------------------------------------------
    # 2. Natureza das Duplicatas
    # -------------------------------------------------------------------------
    print("\n🔍 NATUREZA DAS DUPLICATAS")
    duplicated_mask = df.index.duplicated(keep=False)
    n_dup_idx = duplicated_mask.sum()

    if n_dup_idx > 0:
        dup_df = df[duplicated_mask].copy()
        # Verifica se são exatamente idênticas (todos os valores iguais)
        identical = dup_df.groupby(dup_df.index).apply(
            lambda g: g.drop_duplicates().shape[0] == 1
        )
        n_identical = identical.sum()
        n_conflicting = len(identical) - n_identical

        print(f"  Timestamps duplicados: {len(identical)} únicos")
        print(f"  Duplicatas IDÊNTICAS (mesmos valores): {n_identical}")
        print(f"  Timestamps CONFLITANTES (valores diferentes): {n_conflicting}")

        if n_conflicting > 0:
            print("  ⚠️  ATENÇÃO: existem timestamps com valores conflitantes!")
            print("  Exemplo de conflito:")
            conflict_idx = identical[~identical].index[0]
            print(dup_df.loc[[conflict_idx]].head(6))
        else:
            print("  ✓ Todas as duplicatas são idênticas — podemos remover com segurança.")
    else:
        print("  Nenhuma duplicata encontrada.")

    # -------------------------------------------------------------------------
    # 3. Consistência Física (Lei de Ohm aproximada)
    # -------------------------------------------------------------------------
    print("\n🔍 CONSISTÊNCIA FÍSICA")
    # P = V * I / 1000  (converte W para kW, aproximado)
    df_clean = df.dropna(subset=["Global_active_power", "Voltage", "Global_intensity"])
    calculated_power = (df_clean["Voltage"] * df_clean["Global_intensity"]) / 1000.0
    observed_power = df_clean["Global_active_power"]
    relative_error = np.abs(observed_power - calculated_power) / observed_power

    print(f"  Amostras analisadas: {len(df_clean):,}")
    print(f"  Erro relativo médio: {relative_error.mean():.2%}")
    print(f"  Erro relativo mediano: {relative_error.median():.2%}")
    print(f"  95º percentil do erro: {relative_error.quantile(0.95):.2%}")
    print("  (Valores baixos indicam consistência entre tensão, corrente e potência)")

    # -------------------------------------------------------------------------
    # 4. Outliers Grosseiros (IQR)
    # -------------------------------------------------------------------------
    print("\n🔍 OUTLIERS GROSSEIROS — Global_active_power (IQR method)")
    target = df["Global_active_power"].dropna()
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = target[(target < lower) | (target > upper)]

    print(f"  Limite inferior: {lower:.3f} kW")
    print(f"  Limite superior: {upper:.3f} kW")
    print(f"  Outliers detectados: {len(outliers):,} ({len(outliers)/len(target)*100:.2f}%)")
    print(f"  Valor máximo: {target.max():.3f} kW")
    print(f"  Valor mínimo: {target.min():.3f} kW")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_quality()
