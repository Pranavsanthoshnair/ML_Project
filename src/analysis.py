"""
analysis.py  — COMPLETED IMPLEMENTATION
========================================
Ananthan — Validation, Metrics & Statistical Analysis (visualizations)
Branch: feature/Ananthan-validation

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

CONFIG_ORDER = ["M10", "M20", "M20_I", "M20_R", "M20_I_R", "ALL"]
CONFIG_LABELS = {
    "M10":     "M10\n(10 meaningful)",
    "M20":     "M20\n(20 meaningful)\n[baseline]",
    "M20_I":   "M20+I\n(+irrelevant)",
    "M20_R":   "M20+R\n(+redundant)",
    "M20_I_R": "M20+I+R\n(+both)",
    "ALL":     "ALL\n(all original)",
}
PALETTE     = {"Ridge": "#4C72B0", "Random Forest": "#DD8452"}
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


def _dollar_fmt(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _ordered(df_col):
    return [CONFIG_LABELS.get(c, c) for c in CONFIG_ORDER if c in df_col.values]


def plot_rmse_by_config(results_df: pd.DataFrame, save: bool = True):
    """Plot 1 — Test RMSE by feature configuration."""
    df = results_df.copy()
    df["Label"] = df["Config"].map(CONFIG_LABELS).fillna(df["Config"])
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df, x="Label", y="Test_RMSE", hue="Model",
                order=_ordered(df["Config"]), palette=PALETTE, ax=ax, errorbar=None)
    ax.set_title("Plot 1 — Test RMSE by Feature Configuration\n(M20 is the main baseline)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature Configuration"); ax.set_ylabel("Test RMSE (USD)")
    _dollar_fmt(ax); ax.grid(axis="y", alpha=0.3); ax.legend(title="Model")
    fig.tight_layout()
    if save:
        p = os.path.join(FIGURES_DIR, "plot1_rmse_by_config.png")
        fig.savefig(p, dpi=150); print(f"Saved: {p}")
    plt.show(); return fig


def plot_generalization(results_df: pd.DataFrame, cv_df: pd.DataFrame, save: bool = True):
    """Plot 2 — Train vs CV vs Test RMSE (generalization behaviour)."""
    merged = results_df[["Config","Model","Train_RMSE","Test_RMSE"]].merge(
        cv_df[["Config","Model","CV_RMSE_Mean"]], on=["Config","Model"], how="left")
    plot_df = merged.melt(id_vars=["Config","Model"],
                          value_vars=["Train_RMSE","CV_RMSE_Mean","Test_RMSE"],
                          var_name="Split", value_name="RMSE")
    plot_df["Split"] = plot_df["Split"].map({"Train_RMSE":"Train","CV_RMSE_Mean":"CV (mean)","Test_RMSE":"Test"})
    plot_df["Label"] = plot_df["Config"].map(CONFIG_LABELS).fillna(plot_df["Config"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, mdl in zip(axes, ["Ridge", "Random Forest"]):
        sub = plot_df[plot_df["Model"] == mdl]
        sns.barplot(data=sub, x="Label", y="RMSE", hue="Split",
                    order=_ordered(sub["Config"]),
                    palette={"Train":"#2ca02c","CV (mean)":"#ff7f0e","Test":"#1f77b4"},
                    ax=ax, errorbar=None)
        ax.set_title(f"Plot 2 — {mdl}: Train / CV / Test RMSE", fontweight="bold")
        ax.set_xlabel("Configuration"); ax.set_ylabel("RMSE (USD)")
        _dollar_fmt(ax); ax.tick_params(axis="x", labelsize=8); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Generalization Behaviour", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        p = os.path.join(FIGURES_DIR, "plot2_generalization.png")
        fig.savefig(p, dpi=150); print(f"Saved: {p}")
    plt.show(); return fig


def plot_cv_distribution(cv_df: pd.DataFrame, save: bool = True):
    """Plot 3 — CV RMSE boxplot across folds (stability)."""
    rows = []
    for _, row in cv_df.iterrows():
        for fold_rmse in row["_fold_rmse"]:
            rows.append({"Config": row["Config"], "Model": row["Model"], "CV_RMSE": fold_rmse})
    fold_df = pd.DataFrame(rows)
    fold_df["Label"] = fold_df["Config"].map(CONFIG_LABELS).fillna(fold_df["Config"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, mdl in zip(axes, ["Ridge", "Random Forest"]):
        sub = fold_df[fold_df["Model"] == mdl]
        sns.boxplot(data=sub, x="Label", y="CV_RMSE", order=_ordered(sub["Config"]),
                    color=PALETTE[mdl], width=0.5, ax=ax)
        ax.set_title(f"Plot 3 — {mdl}: CV RMSE Distribution (5 folds)", fontweight="bold")
        ax.set_xlabel("Configuration"); ax.set_ylabel("CV RMSE (USD)")
        _dollar_fmt(ax); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Cross-Validation RMSE Variability", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        p = os.path.join(FIGURES_DIR, "plot3_cv_distribution.png")
        fig.savefig(p, dpi=150); print(f"Saved: {p}")
    plt.show(); return fig


def plot_delta_rmse(delta_df: pd.DataFrame, save: bool = True):
    """Plot 4 — Pairwise delta RMSE (direct research question answer)."""
    main = delta_df[delta_df["Comparison"].str.startswith("M20")].copy()
    col  = "Mean d RMSE" if "Mean d RMSE" in main.columns else "Delta Test RMSE"
    main["Short"] = main["Comparison"].str.replace("M20 → ", "+", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mdl in zip(axes, ["Ridge", "Random Forest"]):
        sub = main[main["Model"] == mdl]
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in sub[col]]
        ax.barh(sub["Short"], sub[col], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Plot 4 — {mdl}: Pairwise ΔRMSE\n(red=worse, green=better)", fontweight="bold")
        ax.set_xlabel("ΔRMSE (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(axis="x", alpha=0.3)
        if "CI Lower" in sub.columns:
            for i, (_, r) in enumerate(sub.iterrows()):
                ax.errorbar(r[col], i,
                            xerr=[[r[col]-r["CI Lower"]], [r["CI Upper"]-r[col]]],
                            fmt="none", color="black", capsize=4)
    fig.suptitle("Does Adding Features Improve Prediction? (vs M20 baseline)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save:
        p = os.path.join(FIGURES_DIR, "plot4_delta_rmse.png")
        fig.savefig(p, dpi=150); print(f"Saved: {p}")
    plt.show(); return fig


def plot_feature_count_vs_rmse(results_df: pd.DataFrame, save: bool = True):
    """Plot 5 — Feature count vs Test RMSE (more features ≠ better)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for mdl, color in PALETTE.items():
        sub = results_df[results_df["Model"] == mdl].sort_values("Raw_Features")
        ax.plot(sub["Raw_Features"], sub["Test_RMSE"], marker="o", linewidth=2, color=color, label=mdl)
        for _, row in sub.iterrows():
            ax.annotate(row["Config"], (row["Raw_Features"], row["Test_RMSE"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)
    ax.set_title("Plot 5 — Feature Count vs Test RMSE\n(More features ≠ better performance)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Raw Features"); ax.set_ylabel("Test RMSE (USD)")
    _dollar_fmt(ax); ax.legend(title="Model"); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        p = os.path.join(FIGURES_DIR, "plot5_feature_count_vs_rmse.png")
        fig.savefig(p, dpi=150); print(f"Saved: {p}")
    plt.show(); return fig


def plot_redundancy_heatmap(X_experiment: pd.DataFrame, redundant_specs: dict, save: bool = True):
    """Heatmap verifying redundant features are near-duplicates."""
    cols = []
    for new_name, (orig, _, _) in redundant_specs.items():
        cols.extend([orig, new_name])
    cols = list(dict.fromkeys(cols))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(X_experiment[cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Redundant Feature Verification: Correlation with Originals\n(Near 1.0 = controlled redundancy)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save:
        p = os.path.join(FIGURES_DIR, "plot_redundancy_heatmap.png")
        fig.savefig(p, dpi=150); print(f"Saved: {p}")
    plt.show(); return fig


def build_evidence_table(results_df, cv_df, delta_df, cv_delta_df) -> pd.DataFrame:
    """Merge all results. Save CSVs to results/tables/."""
    test_cols = ["Config","Model","Raw_Features","Test_RMSE","Test_MAE","Test_R2","Train_RMSE","Generalization_Gap"]
    cv_cols   = ["Config","Model","CV_RMSE_Mean","CV_RMSE_SD","CV_R2_Mean","CV_R2_SD"]
    merged    = results_df[test_cols].merge(cv_df[cv_cols], on=["Config","Model"], how="left")
    merged.round(2).to_csv("results/tables/evidence_table.csv", index=False)
    delta_df.round(2).to_csv("results/tables/pairwise_deltas.csv", index=False)
    cv_delta_df.round(2).to_csv("results/tables/cv_pairwise_deltas.csv", index=False)
    print("Saved: results/tables/evidence_table.csv")
    print("Saved: results/tables/pairwise_deltas.csv")
    print("Saved: results/tables/cv_pairwise_deltas.csv")
    return merged
