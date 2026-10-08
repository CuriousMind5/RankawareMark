# -*- coding: utf-8 -*-


import os
import pandas as pd
import matplotlib.pyplot as plt

# ====================== PATHS ======================
BASE = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_c4_realnewslike_opt-2.7B_entropy_0.9/robustness_tables/Z_score_plots/"

IN_CSV  = BASE + "z_score_stats_attacks_with_translation.csv"
OUT_PNG = BASE + "F1_scatter/f1_vs_tpr_c4_realnewslike_opt-2.7B_entropy_0.9.png"

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ====================== ATTACK LABELS (FOR LEGEND) ======================
# ✅ This is what you asked: legend labels like "Paraphrase (.. attack)" etc.
ATTACK_LABELS = {
    "No Attack": "No Attack",
    "Pegasus":   "Paraphrase (Pegasus attack)",
    "DIPPER":    "Paraphrase (DIPPER attack)",
    "T5-Parrot": "Paraphrase (T5-Parrot attack)",
    "EN-DE-EN":  "Translation (EN-DE-EN)",
    "EN-FR-EN":  "Translation (EN-FR-EN)",
}

# ====================== COLORS (BY ATTACK) ======================
# Colors keyed by RAW attack name (same as CSV)
ATTACK_COLORS = {
    "No Attack": "#1f77b4",
    "Pegasus":   "#2ca02c",
    "DIPPER":    "#ff7f0e",
    "T5-Parrot": "#d62728",
    "EN-DE-EN":  "#9467bd",
    "EN-FR-EN":  "#8c564b",
}

# ====================== MARKERS (BY METHOD) ======================
MARKERS = {
    "KGW": "o",
    "SWEET": "s",
    "Unigram": "P",
    "EWD": "X",
    "Rank-Aware": "D",
}

# ====================== LOAD ======================
if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"Missing CSV: {IN_CSV}")

df = pd.read_csv(IN_CSV)

# Remove Non-WM points (reference plot shows watermark methods only)
df = df[df["method"] != "Non-WM"].copy()

# percent_z>=tau is TPR@5% here (tau fixed)
df["tpr"] = df["percent_z>=tau"] / 100.0

# F1 at fixed 5% FPR -> consistent transform for this scatter style
df["f1"] = (2 * df["tpr"]) / (1 + df["tpr"])

# ====================== PLOT ======================
plt.rcParams.update({
    "figure.figsize": (7.6, 6.4),
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "grid.color": "#EAEAEA",
    "grid.linewidth": 0.8,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
})

fig, ax = plt.subplots()
ax.grid(True, alpha=0.7)

# ====================== SCATTER ======================
for _, r in df.iterrows():
    raw_attack = r["attack"]
    color  = ATTACK_COLORS.get(raw_attack, "#999999")
    marker = MARKERS.get(r["method"], "o")

    ax.scatter(
        r["f1"], r["tpr"],
        s=90,
        color=color,
        marker=marker,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        zorder=3
    )

# ====================== AXES ======================
ax.set_xlabel("F1 Score @ 5% FPR")
ax.set_ylabel("TPR @ 5% FPR")
ax.set_xlim(0.35, 1.01)
ax.set_ylim(0.30, 1.01)
ax.set_title("F1@5% vs TPR@5% Across Attacks")

# ====================== LEGENDS ======================
# Method legend (markers)
method_handles = [
    plt.Line2D([0], [0], marker=mk, linestyle="None",
               color="black", markersize=8, label=method)
    for method, mk in MARKERS.items()
]
leg1 = ax.legend(handles=method_handles, title="Method",
                 loc="upper left", frameon=True)
ax.add_artist(leg1)

# Attack legend (colors) — with your requested descriptive labels
attack_handles = []
for raw_name, color in ATTACK_COLORS.items():
    label = ATTACK_LABELS.get(raw_name, raw_name)
    attack_handles.append(
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=color, markeredgecolor="black",
                   color="w", markersize=8, label=label)
    )

ax.legend(handles=attack_handles, title="Attack Type",
          loc="lower right", frameon=True)

# ====================== SAVE ======================
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close(fig)

print("Saved scatter plot →", OUT_PNG)
