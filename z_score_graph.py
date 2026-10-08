# -*- coding: utf-8 -*-
"""
Z-score Boxplot — MarkLLM Robustness Attacks
Attacks:
  - No Attack
  - Pegasus
  - Dipper
  - T5-Parrot
  - EN-DE-EN
  - EN-FR-EN
"""

from __future__ import annotations

# ======================== TOGGLES ========================
ENABLE_ORIGINAL  = True
ENABLE_PEGASUS   = True
ENABLE_DIPPER    = True
ENABLE_T5PARROT  = True
ENABLE_ENDE_BACK = True   # English -> German -> English
ENABLE_ENFR_BACK = True   # English -> French -> English

# ======================== FILE PATHS ========================
BASE = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"

FILE_ORIGINAL  = BASE + "outputs_c4_realnewslike_opt-6.7B_entropy_0.9/detect_all.jsonl"
FILE_PEGASUS   = BASE + "outputs_c4_realnewslike_opt-6.7B_entropy_0.9/Pegasus/detect_all.jsonl"
FILE_DIPPER    = BASE + "outputs_c4_realnewslike_opt-6.7B_entropy_0.9/Dipper/detect_all.jsonl"
FILE_T5PARROT  = BASE + "outputs_c4_realnewslike_opt-6.7B_entropy_0.9/Parrot/detect_all.jsonl"
FILE_ENDE_BACK = BASE + "outputs_c4_realnewslike_opt-6.7B_entropy_0.9/English-German-English/detect_all.jsonl"
FILE_ENFR_BACK = BASE + "outputs_c4_realnewslike_opt-6.7B_entropy_0.9/English-French-English/detect_all.jsonl"

OUT_DIR = BASE + "outputs_c4_realnewslike_opt-6.7B_entropy_0.9/robustness_tables/Z_score_plots/"
OUT_PNG = OUT_DIR + "z_score_outputs_c4_realnewslike_opt-6.7B_entropy_0.9_CLEAR.png"
OUT_PDF = OUT_DIR + "z_score_outputs_c4_realnewslike_opt-6.7B_entropy_0.9_CLEAR.pdf"
OUT_CSV = OUT_DIR + "z_score_stats_attacks_with_translation.csv"

TAU = 4.0

# ======================== SCHEMES ========================
SCHEMES_ORDER = [
    "vanilla",
    "kgw",
    "sweet",
    "unigram",
    "ewd",
    "rankaware",
]

SCHEME_DISPLAY = {
    "vanilla":   "Non-WM",
    "kgw":       "KGW",
    "sweet":     "SWEET",
    "unigram":   "Unigram",
    "ewd":       "EWD",
    "rankaware": "Rank-Aware",
}

SCHEME_COLORS = {
    "vanilla":   "#7f7f7f",
    "kgw":       "#1f77b4",
    "sweet":     "#2ca02c",
    "unigram":   "#17becf",
    "ewd":       "#8c564b",
    "rankaware": "#d62728",
}

# ======================== IMPORTS ========================
import os
import sys
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ======================== STYLE ========================
plt.rcParams.update({
    "figure.figsize": (22, 11),
    "figure.dpi": 220,
    "savefig.dpi": 450,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 1.0,
    "grid.alpha": 1.0,
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 40,
    "ytick.labelsize": 20,
    "legend.fontsize": 26,
    "legend.title_fontsize": 28,
})

# ======================== HELPERS ========================
def _fail(msg: str) -> None:
    print("[error]", msg)
    sys.exit(1)


def load_zs_by_scheme(path: str):
    d = defaultdict(list)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            s = obj.get("scheme")
            if not s:
                continue

            try:
                z = float(obj.get("z", "nan"))
            except Exception:
                z = float("nan")

            if math.isfinite(z):
                d[s].append(z)

    return d


def stats(zs, tau):
    n = len(zs)
    if n == 0:
        return 0, float("nan"), float("nan"), 0.0

    m = sum(zs) / n
    sd = (sum((z - m) ** 2 for z in zs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    pct = 100.0 * sum(z >= tau for z in zs) / n
    return n, m, sd, pct


def flatten_all_values(all_data, present):
    vals = []
    for d in all_data:
        for s in present:
            vals.extend([x for x in d.get(s, []) if math.isfinite(x)])
    return vals


# ======================== MAIN ========================
def main():
    files = []
    labels = []

    if ENABLE_ORIGINAL:
        files.append(FILE_ORIGINAL)
        labels.append("No Attack")
    if ENABLE_PEGASUS:
        files.append(FILE_PEGASUS)
        labels.append("Pegasus")
    if ENABLE_DIPPER:
        files.append(FILE_DIPPER)
        labels.append("DIPPER")
    if ENABLE_T5PARROT:
        files.append(FILE_T5PARROT)
        labels.append("T5-Parrot")
    if ENABLE_ENDE_BACK:
        files.append(FILE_ENDE_BACK)
        labels.append("EN-DE-EN")
    if ENABLE_ENFR_BACK:
        files.append(FILE_ENFR_BACK)
        labels.append("EN-FR-EN")

    if not files:
        _fail("No dataset enabled.")

    for f in files:
        if not os.path.exists(f):
            _fail(f"Missing file: {f}")

    all_data = [load_zs_by_scheme(f) for f in files]

    present = [s for s in SCHEMES_ORDER if any(s in d for d in all_data)]
    if not present:
        _fail("No schemes found in detect files.")

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots()

    n_files = len(files)
    cluster_w = 0.84
    box_w = cluster_w / max(1, n_files)
    offsets = [(-cluster_w / 2.0) + (i + 0.5) * box_w for i in range(n_files)]
    alphas = [0.35 + 0.55 * (i / max(1, n_files - 1)) for i in range(n_files)]

    # threshold line only
    ax.axhline(
        TAU,
        color="#222222",
        linestyle="--",
        linewidth=2.2,
        zorder=2
    )

    for xi, sch in enumerate(present):
        base = SCHEME_COLORS.get(sch, "#777777")

        for fi in range(n_files):
            zs = all_data[fi].get(sch, [])
            pos = xi + offsets[fi]
            values = zs if zs else [float("nan")]

            ax.boxplot(
                [values],
                positions=[pos],
                widths=box_w * 0.90,
                patch_artist=True,
                showfliers=True,
                whis=1.5,
                flierprops=dict(
                    marker="o",
                    markersize=3.6,
                    markerfacecolor=base,
                    markeredgecolor=base,
                    alpha=alphas[fi],
                    markeredgewidth=0.4
                ),
                boxprops=dict(
                    linewidth=1.8,
                    facecolor=base,
                    alpha=alphas[fi],
                    edgecolor=base
                ),
                whiskerprops=dict(
                    color=base,
                    linewidth=1.6
                ),
                capprops=dict(
                    color=base,
                    linewidth=1.6
                ),
                medianprops=dict(
                    linewidth=2.4,
                    color="#111111"
                ),
                zorder=3
            )

    # x ticks
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([SCHEME_DISPLAY[s] for s in present])

    ax.tick_params(axis="x", pad=10)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")
        label.set_verticalalignment("top")
        label.set_fontsize(24)
        label.set_fontweight("normal")

    for label in ax.get_yticklabels():
        label.set_fontsize(20)

    ax.set_ylabel("Z-score", fontsize=24, labelpad=12)

    # title removed on purpose

    # y-limits
    all_vals = flatten_all_values(all_data, present)
    if all_vals:
        ymin = min(all_vals)
        ymax = max(all_vals)
        yrng = max(1.0, ymax - ymin)

        pad_low = max(2.0, 0.10 * yrng)
        pad_high = max(3.0, 0.12 * yrng)

        ax.set_ylim(ymin - pad_low, ymax + pad_high)

    ax.set_xlim(-0.8, len(present) - 0.2)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # legends INSIDE axes
    method_handles = [
        Patch(
            facecolor=SCHEME_COLORS[s],
            edgecolor=SCHEME_COLORS[s],
            label=SCHEME_DISPLAY[s]
        )
        for s in present
    ]

    attack_handles = [
        Patch(facecolor="black", edgecolor="black", alpha=alphas[i], label=labels[i])
        for i in range(len(labels))
    ]

    # Watermark Method legend -> inside upper left
    leg1 = ax.legend(
        handles=method_handles,
        title="Watermark Method",
        loc="upper left",
        frameon=True,
        borderpad=1.2,
        labelspacing=0.7,
        handlelength=1.6,
        handletextpad=0.8,
        borderaxespad=0.8,
        prop={"size": 26},
        title_fontsize=28
    )
    ax.add_artist(leg1)

    # Attack legend -> inside upper right
    ax.legend(
        handles=attack_handles,
        title="Attack",
        loc="upper right",
        frameon=True,
        borderpad=1.2,
        labelspacing=0.7,
        handlelength=1.6,
        handletextpad=0.8,
        borderaxespad=0.8,
        prop={"size": 26},
        title_fontsize=28
    )

    # complete lines / no clipping
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    plt.savefig(OUT_PDF)
    plt.close(fig)

    print("Saved plot ->", OUT_PNG)
    print("Saved plot ->", OUT_PDF)

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("attack,method,n,mean_z,std_z,percent_z>=tau,tau\n")
        for fi, lab in enumerate(labels):
            for sch in present:
                zs = all_data[fi].get(sch, [])
                n, m, sd, pct = stats(zs, TAU)
                f.write(
                    f"{lab},{SCHEME_DISPLAY[sch]},{n},{m:.6f},{sd:.6f},{pct:.2f},{TAU}\n"
                )

    print("Saved CSV ->", OUT_CSV)


if __name__ == "__main__":
    main()