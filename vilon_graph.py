# -*- coding: utf-8 -*-

import os
import json
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== BASE PATH =====================

BASE_DIR = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/PPL"
)

FILES = {
    "non-watermark": os.path.join(BASE_DIR, "vanilla_n500__ppl_all.jsonl"),
    "kgw":           os.path.join(BASE_DIR, "kgw_n500__ppl_all.jsonl"),
    "sweet":         os.path.join(BASE_DIR, "sweet_n500__ppl_all.jsonl"),
    "unigram":       os.path.join(BASE_DIR, "unigram_n500__ppl_all.jsonl"),
    "ewd":           os.path.join(BASE_DIR, "ewd_n500__ppl_all.jsonl"),
    "rankaware":     os.path.join(BASE_DIR, "rankaware_n500__ppl_all.jsonl"),
}

OUT_PNG = os.path.join(BASE_DIR, "ppl_violin_opt-6.7B_entropy_0.9_updated.png")
OUT_PDF = os.path.join(BASE_DIR, "ppl_violin_opt-6.7B_entropy_0.9_updated.pdf")

MAX_PPL_KEEP = 30.0

# ===================== SCHEME CONFIG =====================

SCHEMES_ORDER = [
    "non-watermark",
    "kgw",
    "sweet",
    "unigram",
    "ewd",
    "rankaware",
]

SCHEME_DISPLAY = {
    "non-watermark": "Unwatermarked",
    "kgw":           "KGW",
    "sweet":         "SWEET",
    "unigram":       "Unigram",
    "ewd":           "EWD",
    "rankaware":     "Rank-Aware",
}

PALETTE = {
    "Unwatermarked": "#9467bd",
    "KGW": "#1f77b4",
    "SWEET": "#2ca02c",
    "Unigram": "#17becf",
    "EWD": "#8c564b",
    "Rank-Aware": "#d62728",
}

# ===================== STYLE =====================

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 22,
    "ytick.labelsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ===================== LOAD =====================

def load_ppl_from_files(files):
    rows = []

    for scheme, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                ppl = obj.get("ppl")
                if ppl is None:
                    continue

                try:
                    ppl = float(ppl)
                except Exception:
                    continue

                if math.isfinite(ppl) and ppl <= MAX_PPL_KEEP:
                    rows.append({
                        "Method": SCHEME_DISPLAY[scheme],
                        "Perplexity": ppl,
                    })

    return pd.DataFrame(rows)

# ===================== MAIN =====================

def main():
    df = load_ppl_from_files(FILES)
    order = [SCHEME_DISPLAY[s] for s in SCHEMES_ORDER]

    fig, ax = plt.subplots(figsize=(14, 6.8))

    sns.violinplot(
        x="Method",
        y="Perplexity",
        data=df,
        order=order,
        palette=PALETTE,
        cut=0,
        inner="box",
        linewidth=1.2,
        saturation=1.0,
        ax=ax,
    )

    ax.set_title("PPL Distribution - All Watermark Methods", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Perplexity", fontsize=18, rotation=90, labelpad=10)

    ax.set_ylim(0, MAX_PPL_KEEP)
    ax.set_yticks(range(0, 31, 5))

    ax.tick_params(axis="x", pad=10, length=0)
    ax.tick_params(axis="y", pad=6)

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")
        label.set_verticalalignment("top")
        label.set_fontsize(18)

    for label in ax.get_yticklabels():
        label.set_fontsize(16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(bottom=0.16, left=0.10, right=0.98, top=0.88)

    plt.savefig(OUT_PNG, dpi=500, bbox_inches="tight")
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")

    print("Saved PNG:", OUT_PNG)
    print("Saved PDF:", OUT_PDF)

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()