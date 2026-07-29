"""
Funções de visualização reutilizáveis.
"""

import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path("figures")


def save_figure(fig, filename: str) -> None:
    """Salva figura em alta resolução."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_vs_real(y_test, y_pred, title: str, filename: str, n_plot: int = 168) -> None:
    """Plota previsão vs real (primeira semana do teste)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test.index[:n_plot], y_test.values[:n_plot], label="Real", color="black", linewidth=1.5)
    ax.plot(y_test.index[:n_plot], y_pred[:n_plot], label="Previsão", color="crimson", linewidth=1.5, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, filename)
