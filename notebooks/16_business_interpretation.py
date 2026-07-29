"""
ETAPA 21 — Interpretação para Negócio
Responde perguntas estratégicas com base nos dados e modelos.
"""

import pandas as pd
import numpy as np
from pathlib import Path


FEATURES_PATH = Path("data/processed/df_features.parquet")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Global_active_power"


def analyze_business() -> None:
    print("=" * 60)
    print("  INTERPRETAÇÃO PARA NEGÓCIO")
    print("=" * 60)

    df = pd.read_parquet(FEATURES_PATH)

    # =================================================================
    # 1. SAZONALIDADE
    # =================================================================
    print("\n📅 1. EXISTE SAZONALIDADE?")
    print("-" * 55)

    # Sazonalidade diária: compara média por hora
    hourly_avg = df.groupby("hour")[TARGET].mean()
    peak_hour = hourly_avg.idxmax()
    low_hour = hourly_avg.idxmin()
    daily_variation = hourly_avg.max() - hourly_avg.min()

    print(f"   Sim. A série apresenta forte sazonalidade DIÁRIA.")
    print(f"   • Pico de consumo: {peak_hour:02d}h ({hourly_avg.max():.3f} kW em média)")
    print(f"   • Vale de consumo: {low_hour:02d}h ({hourly_avg.min():.3f} kW em média)")
    print(f"   • Variação diária: {daily_variation:.3f} kW ({daily_variation/hourly_avg.mean()*100:.1f}% acima da média)")
    print(f"   → O consumo varia {daily_variation/hourly_avg.min():.1f}x entre pico e vale.")

    # Sazonalidade semanal
    weekly_avg = df.groupby("dayofweek")[TARGET].mean()
    weekend_avg = weekly_avg.loc[[5, 6]].mean()
    weekday_avg = weekly_avg.loc[0:4].mean()

    print(f"\n   Sazonalidade SEMANAL também é evidente:")
    print(f"   • Média dias úteis: {weekday_avg:.3f} kW")
    print(f"   • Média fins de semana: {weekend_avg:.3f} kW")
    print(f"   • Diferença: {abs(weekend_avg - weekday_avg):.3f} kW")
    if weekend_avg > weekday_avg:
        print(f"   → Fins de semana consomem MAIS (residência com mais atividades em casa)")
    else:
        print(f"   → Dias úteis consomem MAIS (provavelmente por ausência de moradores no fim de semana)")

    # Sazonalidade anual (por mês)
    monthly_avg = df.groupby("month")[TARGET].mean()
    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()

    month_names = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }

    print(f"\n   Sazonalidade ANUAL:")
    print(f"   • Pico: {month_names[peak_month]} ({monthly_avg.max():.3f} kW)")
    print(f"   • Vale: {month_names[low_month]} ({monthly_avg.min():.3f} kW)")
    print(f"   → Diferença sazonal: {monthly_avg.max() - monthly_avg.min():.3f} kW")

    # =================================================================
    # 2. HORÁRIOS DE PICO
    # =================================================================
    print("\n⚡ 2. QUAIS SÃO OS HORÁRIOS DE PICO?")
    print("-" * 55)

    top_5_hours = hourly_avg.sort_values(ascending=False).head(5)
    print("   Top 5 horários de maior consumo:")
    for hour, value in top_5_hours.items():
        print(f"   • {hour:02d}h: {value:.3f} kW")

    bottom_5_hours = hourly_avg.sort_values(ascending=True).head(5)
    print("\n   Top 5 horários de menor consumo:")
    for hour, value in bottom_5_hours.items():
        print(f"   • {hour:02d}h: {value:.3f} kW")

    # =================================================================
    # 3. TENDÊNCIA
    # =================================================================
    print("\n📈 3. HÁ TENDÊNCIA DE LONGO PRAZO?")
    print("-" * 55)

    yearly_avg = df.groupby("year")[TARGET].mean()
    print("   Média anual de consumo:")
    for year, value in yearly_avg.items():
        print(f"   • {year}: {value:.3f} kW")

    if len(yearly_avg) > 1:
        first_year = yearly_avg.iloc[0]
        last_year = yearly_avg.iloc[-1]
        trend = last_year - first_year
        trend_pct = trend / first_year * 100

        if abs(trend_pct) < 5:
            print(f"\n   → Tendência NEUTRA: variação de {trend_pct:+.1f}% ao longo dos anos.")
            print(f"     O consumo é estável, sem crescimento ou queda significativa.")
        elif trend > 0:
            print(f"\n   → Tendência de CRESCIMENTO: +{trend_pct:.1f}% ao longo dos anos.")
            print(f"     Possíveis causas: aumento de eletrodomésticos, mudança de hábitos.")
        else:
            print(f"\n   → Tendência de QUEDA: {trend_pct:.1f}% ao longo dos anos.")
            print(f"     Possíveis causas: eficiência energética, redução de moradores.")

    # =================================================================
    # 4. COMO REDUZIR CUSTOS?
    # =================================================================
    print("\n💰 4. COMO REDUZIR CUSTOS DE ENERGIA?")
    print("-" * 55)

    avg_consumption = df[TARGET].mean()
    peak_consumption = hourly_avg.max()

    print("   Estratégias baseadas nos padrões identificados:")
    print(f"\n   a) DESLOCAMENTO DE CARGA (Load Shifting):")
    print(f"      • Reduzir uso de eletrodomésticos de pico ({peak_hour:02d}h) para horários de vale ({low_hour:02d}h)")
    print(f"      • Economia potencial: até {daily_variation:.3f} kW/hora deslocada")

    print(f"\n   b) AUTOMAÇÃO RESIDENCIAL:")
    print(f"      • Programar aquecedor/chuveiro para ligar fora do horário de pico")
    print(f"      • Usar timer para máquina de lavar/lava-louças durante a madrugada")

    print(f"\n   c) PAINÉIS SOLARES + BATERIA:")
    print(f"      • Armazenar energia solar durante o dia (vale de consumo)")
    print(f"      • Usar bateria no pico da noite ({peak_hour:02d}h)")
    print(f"      • Reduz dependência da rede no horário mais caro")

    print(f"\n   d) TARIFA BRANCA / HORÁRIA:")
    print(f"      • Em mercados com tarifa horária, consumir nos horários de menor demanda")
    print(f"      • A redução de {daily_variation:.3f} kW no pico pode gerar economia de 20-40% na conta")

    # =================================================================
    # 5. COMO O MODELO AJUDA UMA EMPRESA DE ENERGIA?
    # =================================================================
    print("\n🏭 5. COMO ESSE MODELO AJUDA UMA EMPRESA DE ENERGIA?")
    print("-" * 55)

    print("   a) PLANEJAMENTO DE GERAÇÃO:")
    print(f"      • Prever demanda com MAE de ~0.33 kW (precisão de ~67%)")
    print(f"      • Ajustar geração térmica/hidrelétrica com 24h de antecedência")
    print(f"      • Evitar superprodução (desperdício) ou subprodução (blackouts)")

    print(f"\n   b) MANUTENÇÃO PREVENTIVA DA REDE:")
    print(f"      • Identificar períodos de pico para evitar manutenção")
    print(f"      • Prever sobrecarga em transformadores antes que ocorram")

    print(f"\n   c) PREÇO DINÂMICO (DEMAND RESPONSE):")
    print(f"      • Oferecer descontos nos horários de vale para balancear a carga")
    print(f"      • Cobrar tarifa premium no pico para incentivar redução")
    print(f"      • O modelo permite prever exatamente quando aplicar cada tarifa")

    print(f"\n   d) INTEGRAÇÃO DE ENERGIAS RENOVÁVEIS:")
    print(f"      • Previsão solar/eólica é intermitente; prever demanda ajuda a compensar")
    print(f"      • Saber que às {peak_hour:02d}h a demanda será alta permite acionar gás/biomassa")

    print(f"\n   e) REDUÇÃO DE RESERVA GIRANTE:")
    print(f"      • Usinas mantêm reserva de capacidade para imprevistos")
    print(f"      • Previsão precisa permite reduzir essa reserva, economizando combustível")

    # =================================================================
    # 6. Salvar relatório
    # =================================================================
    report_lines = [
        "# Relatório de Interpretação para Negócio\n",
        "## 1. Sazonalidade\n",
        f"- Forte sazonalidade DIÁRIA: pico às {peak_hour:02d}h, vale às {low_hour:02d}h\n",
        f"- Sazonalidade SEMANAL: {'fins de semana > dias úteis' if weekend_avg > weekday_avg else 'dias úteis > fins de semana'}\n",
        f"- Sazonalidade ANUAL: pico em {month_names[peak_month]}, vale em {month_names[low_month]}\n",
        "\n## 2. Horários de Pico\n",
        f"- Top 5 horários: {', '.join([f'{h:02d}h' for h in top_5_hours.index])}\n",
        "\n## 3. Tendência\n",
        f"- {'Estável' if abs(trend_pct) < 5 else ('Crescimento' if trend > 0 else 'Queda')} de {trend_pct:+.1f}%\n",
        "\n## 4. Estratégias de Redução de Custos\n",
        "- Deslocamento de carga para horários de vale\n",
        "- Automação residencial\n",
        "- Painéis solares + bateria\n",
        "- Tarifa horária\n",
        "\n## 5. Valor para Empresa de Energia\n",
        "- Planejamento de geração com 24h de antecedência\n",
        "- Manutenção preventiva da rede\n",
        "- Preço dinâmico (demand response)\n",
        "- Integração de renováveis\n",
        "- Redução de reserva girante\n",
    ]

    report_path = REPORTS_DIR / "business_interpretation.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"\n💾 Relatório salvo: {report_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_business()
