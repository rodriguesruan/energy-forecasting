# Energy Consumption Forecasting

Previsao de consumo de energia eletrica residencial usando series temporais e Machine Learning.

## Problema de Negocio

Empresas de energia eletrica precisam equilibrar oferta e demanda em tempo real. Erros de previsao resultam em superproducao (desperdicio), subproducao (blackouts) e custos operacionais elevados.

## Objetivo

Construir um pipeline de previsao de consumo de energia capaz de identificar padroes sazonais, comparar modelos estatisticos com ML, gerar intervalos de confianca e produzir insights acionaveis.

## Dataset

- Fonte: UCI Machine Learning Repository - Household Power Consumption
- Periodo: Dezembro/2006 a Novembro/2010 (~4 anos)
- Granularidade: Minuto a minuto agregado para hora
- Registros: 2.075.259 (raw) -> 34.158 (horario)
- Target: Global_active_power (kW)

## Tecnologias

- Python 3.12
- Pandas, NumPy
- Matplotlib, Seaborn
- Statsmodels (ADF, SARIMA)
- Scikit-learn, XGBoost
- VS Code + WSL (Ubuntu) + venv
- Git + GitHub

## Metodologia

ETAPA 1  -> Planejamento
ETAPA 2  -> Ambiente virtual (.venv)
ETAPA 3  -> Dependencias (requirements.txt)
ETAPA 4  -> Dataset (UCI)
ETAPA 5  -> Estrutura de pastas
ETAPA 6  -> Imports e verificacao
ETAPA 7  -> Carga de dados
ETAPA 8  -> Inspecao inicial
ETAPA 9  -> Analise de qualidade
ETAPA 10 -> Limpeza
ETAPA 11 -> Feature Engineering
ETAPA 12 -> EDA
ETAPA 13 -> Estacionariedade (ADF)
ETAPA 14 -> Split temporal
ETAPA 15 -> Baselines
ETAPA 16 -> ARIMA / SARIMA
ETAPA 17 -> Machine Learning
ETAPA 18 -> Avaliacao
ETAPA 19 -> Visualizacao
ETAPA 20 -> Intervalo de Confianca
ETAPA 21 -> Interpretacao de Negocio
ETAPA 22 -> Organizacao do Codigo
ETAPA 23 -> README Profissional
ETAPA 24 -> GitHub

## Resultados

### Ranking dos Modelos (ordenado por MAE)

| Modelo            | MAE (kW) | RMSE (kW) | MAPE (%) | R2     |
|-------------------|----------|-----------|----------|--------|
| XGBoost           | 0.3363   | 0.4843    | 43.47    | 0.5762 |
| Random Forest     | 0.3396   | 0.4904    | 44.49    | 0.5654 |
| Naive             | 0.6096   | 0.7453    | 93.63    | -0.0036|
| Media Historica   | 0.6350   | 0.7498    | 109.57   | -0.0160|
| Media Movel (24h) | 0.7629   | 0.8834    | 158.39   | -0.4100|

XGBoost reduziu o erro em 47% em relacao ao melhor baseline.

### Insights de Negocio

- Sazonalidade DIARIA: pico 20h (1.90 kW), vale 04h (0.44 kW)
- Sazonalidade SEMANAL: fins de semana > dias uteis
- Sazonalidade ANUAL: dezembro > agosto
- TENDENCIA: queda de 44% de 2006 para 2010 (eficiencia energetica)
- Horarios de pico: 20h, 21h, 19h

## Metricas (XGBoost)

| Metrica | Valor    |
|---------|----------|
| MAE     | 0.336 kW |
| RMSE    | 0.484 kW |
| MAPE    | 43.47%   |
| R2      | 57.62%   |

## Conclusao

1. Series temporais de consumo de energia sao altamente previsiveis
2. ML supera modelos classicos com features temporais e lag features adequadas
3. Lag de 1 hora e o preditor mais importante (62% de importancia)
4. Intervalos de confianca sao essenciais para decisoes de negocio
5. Data leakage e o erro mais comum em forecasting com ML

## Proximos Passos

- Validacao temporal (TimeSeriesSplit)
- Deep Learning (LSTM, Transformer)
- Feriados e dados climaticos
- Deploy via API (FastAPI)
- Otimizacao de hiperparametros (Optuna)

## Como Reproduzir

```bash
git clone https://github.com/seu-usuario/energy-forecasting.git
cd energy-forecasting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
Estrutura do Projeto
plain
energy-forecasting/
├── data/raw/
├── data/processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── visualization/
├── figures/
├── reports/
├── main.py
├── requirements.txt
└── README.md
Licenca
MIT License
Autor: Ruan | Data: 2026 | Status: Concluido
