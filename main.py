"""
Pipeline principal do projeto Energy Forecasting.
Executa todo o fluxo: carga → preprocessamento → modelagem → avaliação.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.load_data import load_dataset
from src.data.preprocess import clean_and_aggregate, create_features, temporal_split
from src.evaluation.metrics import calculate_metrics
from src.models.baselines import naive_forecast, moving_average_forecast, historical_mean_forecast
from src.models.ml_models import prepare_ml_features, train_random_forest, train_xgboost
from src.visualization.plots import plot_prediction_vs_real


TARGET = "Global_active_power"
LAGS = [1, 2, 24, 168]


def main() -> None:
    print("=" * 60)
    print("  ENERGY FORECASTING — PIPELINE COMPLETO")
    print("=" * 60)

    # -----------------------------------------------------------------
    # 1. Carga
    # -----------------------------------------------------------------
    print("\n📥 Carregando dados...")
    df_raw = load_dataset()
    print(f"   Raw: {df_raw.shape[0]:,} registros")

    # -----------------------------------------------------------------
    # 2. Pré-processamento
    # -----------------------------------------------------------------
    print("\n🔧 Pré-processando...")
    df_clean = clean_and_aggregate(df_raw)
    df_features = create_features(df_clean, target=TARGET, lags=LAGS)
    print(f"   Features: {df_features.shape[0]:,} registros × {df_features.shape[1]} colunas")

    # -----------------------------------------------------------------
    # 3. Split
    # -----------------------------------------------------------------
    print("\n✂️  Dividindo treino/teste...")
    df_train, df_test = temporal_split(df_features, test_size=0.20)
    print(f"   Treino: {len(df_train):,} | Teste: {len(df_test):,}")

    # -----------------------------------------------------------------
    # 4. Baselines
    # -----------------------------------------------------------------
    print("\n📊 Baselines...")
    y_train = df_train[TARGET]
    y_test = df_test[TARGET]

    pred_naive = naive_forecast(y_train, len(y_test))
    pred_ma = moving_average_forecast(y_train, len(y_test))
    pred_mean = historical_mean_forecast(y_train, len(y_test))

    metrics_naive = calculate_metrics(y_test, pred_naive)
    metrics_ma = calculate_metrics(y_test, pred_ma)
    metrics_mean = calculate_metrics(y_test, pred_mean)

    print(f"   Naive — MAE: {metrics_naive['MAE']}, RMSE: {metrics_naive['RMSE']}, MAPE: {metrics_naive['MAPE']}%")
    print(f"   MA(24h) — MAE: {metrics_ma['MAE']}, RMSE: {metrics_ma['RMSE']}, MAPE: {metrics_ma['MAPE']}%")
    print(f"   Média Hist — MAE: {metrics_mean['MAE']}, RMSE: {metrics_mean['RMSE']}, MAPE: {metrics_mean['MAPE']}%")

    # -----------------------------------------------------------------
    # 5. Machine Learning
    # -----------------------------------------------------------------
    print("\n🚀 Machine Learning...")
    X_train, y_train_ml = prepare_ml_features(df_train, TARGET)
    X_test, y_test_ml = prepare_ml_features(df_test, TARGET)

    print(f"   ML features: {X_train.shape[1]} | Treino: {len(X_train):,} | Teste: {len(X_test):,}")

    rf = train_random_forest(X_train, y_train_ml)
    xgb_model = train_xgboost(X_train, y_train_ml)

    pred_rf = rf.predict(X_test)
    pred_xgb = xgb_model.predict(X_test)

    metrics_rf = calculate_metrics(y_test_ml, pred_rf)
    metrics_xgb = calculate_metrics(y_test_ml, pred_xgb)

    print(f"   Random Forest — MAE: {metrics_rf['MAE']}, RMSE: {metrics_rf['RMSE']}, MAPE: {metrics_rf['MAPE']}%")
    print(f"   XGBoost — MAE: {metrics_xgb['MAE']}, RMSE: {metrics_xgb['RMSE']}, MAPE: {metrics_xgb['MAPE']}%")

    # -----------------------------------------------------------------
    # 6. Resumo
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESUMO FINAL")
    print("=" * 60)
    summary = pd.DataFrame({
        "Naive": metrics_naive,
        "MA(24h)": metrics_ma,
        "Média Hist": metrics_mean,
        "Random Forest": metrics_rf,
        "XGBoost": metrics_xgb,
    }).T
    summary = summary.sort_values("MAE")
    print(summary.to_string())

    # -----------------------------------------------------------------
    # 7. Visualização
    # -----------------------------------------------------------------
    print("\n📈 Gerando gráficos...")
    plot_prediction_vs_real(y_test_ml, pred_xgb, "XGBoost — Previsão vs Real", "main_xgboost_prediction.png")
    print("   ✅ figures/main_xgboost_prediction.png")

    print("\n" + "=" * 60)
    print("  PIPELINE CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()
