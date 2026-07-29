"""
ETAPA 6 — Importação e Verificação de Bibliotecas
Verifica se todas as dependências estão corretamente instaladas
e configura o ambiente de visualização.
"""

# =============================================================================
# 1. Manipulação de dados
# =============================================================================
import pandas as pd
import numpy as np

# =============================================================================
# 2. Visualização
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 3. Séries Temporais e Estatística
# =============================================================================
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =============================================================================
# 4. Machine Learning
# =============================================================================
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# =============================================================================
# 5. Utilitários
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 6. Configurações de visualização
# =============================================================================
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

# =============================================================================
# 7. Verificação
# =============================================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  Todas as bibliotecas importadas com sucesso!")
    print("=" * 55)
    print(f"  Pandas        : {pd.__version__}")
    print(f"  NumPy         : {np.__version__}")
    print(f"  Matplotlib    : {plt.matplotlib.__version__}")
    print(f"  Seaborn       : {sns.__version__}")
    print(f"  Statsmodels   : {statsmodels.__version__}")
    print(f"  Scikit-learn  : {sklearn.__version__}")
    print(f"  XGBoost       : {xgb.__version__}")
    print("=" * 55)
