# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ===================== BASE PATH =====================

BASE_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_c4_realnewslike_opt-6.7_entropy_0.9/rank_decay"

RANKS = [20, 40, 60, 80, 100, 120, 140, 160, 180]

OUT_PNG = os.path.join(BASE_DIR, "rank_decay_ppl_bar.png")
OUT_PDF = os.path.join(BASE_DIR, "rank_decay_ppl_bar.pdf")

means = []
stds = []

# ===================== LOAD PPL =====================

for r in RANKS:

    ppl_dir = os.path.join(BASE_DIR, f"vanilla_vs_rank{r}", "PPL")

    if not os.path.exists(ppl_dir):
        means.append(np.nan)
        stds.append(np.nan)
        continue

    files = [f for f in os.listdir(ppl_dir) if "__ppl_all.jsonl" in f]

    if not files:
        means.append(np.nan)
        stds.append(np.nan)
        continue

    ppl_path = os.path.join(ppl_dir, files[0])

    vals = []

    with open(ppl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if "ppl" in obj:
                try:
                    vals.append(float(obj["ppl"]))
                except Exception:
                    continue

    vals = np.array(vals, dtype=float)

    if vals.size == 0:
        means.append(np.nan)
        stds.append(np.nan)
    else:
        means.append(vals.mean())
        stds.append(vals.std())

# ===================== PLOT =====================

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

plt.figure(figsize=(7, 4))

x = np.arange(len(RANKS))

# ---- Soft pastel gradient ----
cmap = cm.get_cmap("Spectral")
colors = cmap(np.linspace(0.1, 0.9, len(RANKS)))

plt.bar(
    x,
    means,
    yerr=stds,
    capsize=4,
    color=colors,
    edgecolor="black",
    linewidth=0.8
)

plt.xticks(x, RANKS)
plt.xlabel("Rank Decay")
plt.ylabel("Mean Perplexity")
plt.title("Effect of Rank Decay on Text Quality (PPL)")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
plt.close()

print("\nSaved PNG:", OUT_PNG)
print("Saved PDF:", OUT_PDF)