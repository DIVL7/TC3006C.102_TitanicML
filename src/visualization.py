from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .utils import p, ensure_dirs, save_caption

# Estilo consistente y accesible
sns.set_theme(style="whitegrid", context="notebook")
CB_PALETTE = sns.color_palette("colorblind")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.titlepad": 10,
})

def _save(fig: plt.Figure, name: str, caption: str) -> Path:
    ensure_dirs()
    out = p("results", "figures", f"{name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    save_caption(out, caption)
    plt.close(fig)
    return out

def plot_hist(df: pd.DataFrame, col: str, bins: int, name: str, title: str, xlabel: str, caption: str):
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), bins=bins, ax=ax, color=CB_PALETTE[0], edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    return _save(fig, name, caption)

def plot_bar_counts(df: pd.DataFrame, col: str, order=None, name: str = "", title: str = "", xlabel: str = "", caption: str = ""):
    fig, ax = plt.subplots()
    vc = df[col].value_counts(dropna=False)
    if order is not None:
        vc = vc.reindex(order)
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette=CB_PALETTE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Conteo")
    return _save(fig, name, caption)

def plot_rate_bar(series: pd.Series, name: str, title: str, xlabel: str, ylabel: str, caption: str, order=None):
    fig, ax = plt.subplots()
    s = series.copy()
    if order is not None:
        s = s.reindex(order)
    sns.barplot(x=s.index.astype(str), y=s.values, ax=ax, palette=CB_PALETTE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    return _save(fig, name, caption)

def plot_grouped_rate_bars(pivot_df: pd.DataFrame, name: str, title: str, xlabel: str, ylabel: str, caption: str):
    """
    Barras agrupadas para tasas (0-1). index = categorías, columns = subgrupos.
    """
    fig, ax = plt.subplots()
    pivot_df = pivot_df.copy()
    pivot_df = pivot_df.loc[:, [c for c in pivot_df.columns if c is not None]]  # limpia columnas None
    pivot_df.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5, color=CB_PALETTE[:len(pivot_df.columns)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    # etiquetas arriba de cada barra
    for p in ax.patches:
        if p.get_height() is not None and not np.isnan(p.get_height()):
            ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.02, f"{p.get_height():.0%}",
                    ha="center", va="bottom", fontsize=8, rotation=0)
    ax.legend(title=pivot_df.columns.name if pivot_df.columns.name else "Grupo")
    return _save(fig, name, caption)

def plot_survival_heatmap(pivot_df: pd.DataFrame, name: str, title: str, xlabel: str, ylabel: str, caption: str):
    """
    Heatmap para tasas (0-1). pivot_df: index x columns -> valores en [0,1].
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot_df, ax=ax, cmap="crest", vmin=0, vmax=1,
                annot=True, fmt=".0%", cbar_kws={"shrink": .8})
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save(fig, name, caption)

def plot_corr_heatmap(df: pd.DataFrame, name: str, title: str, caption: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, annot=False, square=True, cbar_kws={"shrink": .8})
    ax.set_title(title)
    return _save(fig, name, caption)

def plot_roc_curves(fprs_tprs: dict[str, tuple[np.ndarray, np.ndarray, float]], name: str, title: str, caption: str):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, (label, (fpr, tpr, auc_val)) in enumerate(fprs_tprs.items()):
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})", linewidth=2, color=CB_PALETTE[i % len(CB_PALETTE)])
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, name, caption)

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], name: str, title: str, caption: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="crest", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    return _save(fig, name, caption)

from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_true, y_proba, name: str, title: str, caption: str):
    """Curva Precision–Recall (muestra AP en la leyenda)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend([f"AP = {ap:.3f}"], loc="lower left")
    return _save(fig, name, caption)

def plot_threshold_tradeoff(df_thr: pd.DataFrame, name: str, title: str, caption: str):
    """
    Dibuja precision y recall vs threshold para visualizar trade-offs.
    Espera columnas: 'threshold','precision','recall'
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(df_thr["threshold"], df_thr["precision"], label="Precision", linewidth=2)
    ax.plot(df_thr["threshold"], df_thr["recall"], label="Recall", linewidth=2)
    ax.plot(df_thr["threshold"], df_thr["specificity"], label="Specificity (Recall clase 0)", linewidth=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="best")
    return _save(fig, name, caption)

# SHAP y barras de importancias
import shap

def plot_shap_summary(shap_values, X_display, name: str, title: str, caption: str):
    """Fig X: SHAP summary beeswarm."""
    fig = plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, show=False, max_display=25)
    plt.title(title)
    return _save(fig, name, caption)

def plot_shap_waterfall(shap_values_row, name: str, title: str, caption: str):
    """Fig Z: SHAP waterfall para un caso."""
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values_row, show=False, max_display=20)
    plt.title(title)
    return _save(fig, name, caption)

def plot_importances_comparison(df_long: pd.DataFrame, name: str, title: str, caption: str, top_k=10):
    """
    Fig Y: Barras horizontales comparando top_k importancias por modelo.
    Espera columnas: feature, importance, model.
    """
    # top_k por modelo
    tops = (df_long.sort_values(["model","importance"], ascending=[True, False])
                  .groupby("model", group_keys=False).head(top_k))
    # para legibilidad, orden por importancia dentro de cada modelo
    g = tops.copy()
    g["feat_label"] = g["feature"].astype(str)
    # Usamos FacetGrid-like manual (una columna por modelo)
    models = list(g["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(6*n, 6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        sub = g[g["model"]==m].sort_values("importance", ascending=True)
        ax.barh(sub["feat_label"], sub["importance"], color=CB_PALETTE[0])
        ax.set_title(m)
        ax.set_xlabel("Importancia (normalizada)")
        ax.set_ylabel("")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, name, caption)
