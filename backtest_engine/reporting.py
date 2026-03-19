from __future__ import annotations

import matplotlib

# Use non-interactive backend to allow plotting in worker threads.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_param_heatmap(df: pd.DataFrame, value_col: str, row: str, col: str, title: str = "Parameter Heatmap"):
    """
    Create a heatmap of performance metric across two parameter dimensions.

    Expected columns: [row, col, value_col].
    """
    pivot = df.pivot(index=row, columns=col, values=value_col)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(title)
    plt.ylabel(row)
    plt.xlabel(col)
    plt.tight_layout()
    return plt.gcf()
