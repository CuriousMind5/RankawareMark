# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== PATH CONFIG =====================

BASE_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_c4_realnewslike_opt-6.7"

CONFIGS = [
    ("0.6", "Delta_2.0_0.6_60"),
    ("0.9", "Delta_2.0_0.9_60"),
    ("1.2", "Delta_2.0_1.2_60"),
]

PPL_SUBDIR = "PPL"

FILES = {
    "SWEET": "sweet_n500__ppl_all.jsonl",
    "Rank-Aware": "rankaware_n500__ppl_all.jsonl",
}

OUT_PNG = os.path.join(BASE_DIR, "sweet_rankaware_mean_ppl_bar.png")
OUT_PDF = os.path.join(BASE_DIR, "sweet_rankaware_mean_ppl_bar.pdf")

# ===================== FILTER SETTINGS =====================

MAX_PPL_KEEP = 30.0
USE_PERCENTILE = False
PERCENTILE = 99

# ===================== COLORS =====================

COLORS = {
    "sweet_fill": "#7BC96F",
    "sweet_edge": "#238B45",
    "rank_fill": "#76D7EA",
    "rank_edge": "#1BA3C6",
}

# ===================== LOADER =====================

def load_filtered_ppl(path):
    vals = []

    with open(path, "rb") as f:
        for raw in f:
            try:
                obj = json.loads(raw.decode("utf-8", errors="ignore"))
                if "ppl" in obj:
                    v = float(obj["ppl"])
                    if np.isfinite(v):
                        vals.append(v)
            except Exception:
                continue

    vals = np.array(vals, dtype=float)

    if len(vals) == 0:
        return vals

    # --- robust filtering ---
    if USE_PERCENTILE:
        cap = np.percentile(vals, PERCENTILE)
        vals = vals[vals <= cap]
    else:
        vals = vals[vals <= MAX_PPL_KEEP]

    return vals

# ===================== AGGREGATION =====================

sweet_mean, sweet_std = [], []
rank_mean, rank_std = [], []

print("\n===== Mean PPL Aggregation =====")

for tau, folder in CONFIGS:
    base = os.path.join(BASE_DIR, folder, PPL_SUBDIR)

    sweet_vals = load_filtered_ppl(os.path.join(base, FILES["SWEET"]))
    rank_vals  = load_filtered_ppl(os.path.join(base, FILES["Rank-Aware"]))

    s_mean = np.nanmean(sweet_vals)
    s_std  = np.nanstd(sweet_vals)

    r_mean = np.nanmean(rank_vals)
    r_std  = np.nanstd(rank_vals)

    sweet_mean.append(s_mean)
    sweet_std.append(s_std)

    rank_mean.append(r_mean)
    rank_std.append(r_std)

    print(f"\nTau = {tau}")
    print("SWEET mean/std:", s_mean, s_std)
    print("RANK  mean/std:", r_mean, r_std)

# ===================== BAR PLOT =====================

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(CONFIGS))
width = 0.35

plt.figure(figsize=(7, 4))

plt.bar(
    x - width / 2,
    sweet_mean,
    width,
    yerr=sweet_std,
    capsize=4,
    label="SWEET",
    color=COLORS["sweet_fill"],
    edgecolor=COLORS["sweet_edge"],
    linewidth=1.2,
)

plt.bar(
    x + width / 2,
    rank_mean,
    width,
    yerr=rank_std,
    capsize=4,
    label="Rank-Aware",
    color=COLORS["rank_fill"],
    edgecolor=COLORS["rank_edge"],
    linewidth=1.2,
)

plt.xticks(x, [f"t={tau}" for tau, _ in CONFIGS])
plt.ylabel("Mean Perplexity")
plt.title("Mean PPL vs Entropy Threshold")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
plt.close()

print("\nMean PPL bar plot saved to PNG:", OUT_PNG)
print("Mean PPL bar plot saved to PDF:", OUT_PDF)