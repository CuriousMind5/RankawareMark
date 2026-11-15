# -*- coding: utf-8 -*-


from __future__ import annotations

# ======================== TOGGLES ========================
ENABLE_ORIGINAL  = True
ENABLE_PEGASUS   = True
ENABLE_DIPPER    = True
ENABLE_T5PARROT  = True
ENABLE_ENDE_BACK = True
ENABLE_ENFR_BACK = True

# ======================== FILE PATHS ========================
BASE = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"

FILE_ORIGINAL  = BASE + "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/detect_all.jsonl"
FILE_PEGASUS   = BASE + "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/Pegasus/detect_all.jsonl"
FILE_DIPPER    = BASE + "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/Dipper/detect_all.jsonl"
FILE_T5PARROT  = BASE + "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/Parrot/detect_all.jsonl"

FILE_ENDE_BACK = BASE + "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/English-German-English/detect_all.jsonl"
FILE_ENFR_BACK = BASE + "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/English-French-English/detect_all.jsonl"

OUT_DIR = BASE + "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/robustness_tables/roc_plots_each_attack/"

# ======================== SCHEMES ========================
NEGATIVE_SCHEME = "vanilla"

METHODS = [
    ("kgw",       "KGW"),
    ("sweet",     "SWEET"),
    ("unigram",   "Unigram"),
    ("ewd",       "EWD"),
    ("rankaware", "Rank-Aware"),
]

# ======================== IMPORTS ========================
import os
import sys
import json
import math
import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (8, 8),
    "figure.dpi": 300,
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.9,
    "font.size": 12,
})

# ======================== HELPERS ========================
def _fail(msg):
    print("[error]", msg)
    sys.exit(1)


def safe_filename(name):
    
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def enabled_tasks():
    out = []

    if ENABLE_ORIGINAL:
        out.append(("No Attack", FILE_ORIGINAL))

    if ENABLE_PEGASUS:
        out.append(("Pegasus", FILE_PEGASUS))

    if ENABLE_DIPPER:
        out.append(("Dipper", FILE_DIPPER))

    if ENABLE_T5PARROT:
        out.append(("T5-Parrot", FILE_T5PARROT))

    if ENABLE_ENDE_BACK:
        out.append(("EN-DE-EN", FILE_ENDE_BACK))

    if ENABLE_ENFR_BACK:
        out.append(("EN-FR-EN", FILE_ENFR_BACK))

    if not out:
        _fail("No attack files are enabled.")

    for attack_name, path in out:
        if not os.path.exists(path):
            _fail(f"Missing file for {attack_name}: {path}")

    return out


def load_z(path, scheme):
    zs = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            try:
                obj = json.loads(ln)

                if obj.get("scheme") == scheme:
                    z = float(obj.get("z", "nan"))

                    if math.isfinite(z):
                        zs.append(z)

            except Exception:
                pass

    return zs


def roc_points(pos, neg):
    if not pos or not neg:
        return [], float("nan")

    uniq = sorted(set(pos + neg))
    thr = [-1e9] + uniq + [1e9]

    P = len(pos)
    N = len(neg)

    pos_sorted = sorted(pos)
    neg_sorted = sorted(neg)

    def ge(a, t):
        lo, hi = 0, len(a)

        while lo < hi:
            mid = (lo + hi) // 2

            if a[mid] >= t:
                hi = mid
            else:
                lo = mid + 1

        return len(a) - lo

    pts = []

    for t in thr:
        fpr = ge(neg_sorted, t) / N
        tpr = ge(pos_sorted, t) / P
        pts.append((fpr, tpr))

    pts.sort()

    dedup = [pts[0]]

    for x, y in pts[1:]:
        if abs(x - dedup[-1][0]) < 1e-12:
            dedup[-1] = (dedup[-1][0], max(dedup[-1][1], y))
        else:
            dedup.append((x, y))

    auc = 0.0

    for i in range(1, len(dedup)):
        x0, y0 = dedup[i - 1]
        x1, y1 = dedup[i]
        auc += (x1 - x0) * (y0 + y1) * 0.5

    return dedup, auc


def smooth(points, n=600):
    if not points:
        return [], []

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    grid = np.linspace(0, 1, n)

    uniq_x, idx = np.unique(xs, return_index=True)
    uniq_y = ys[idx]

    smoothed = np.interp(grid, uniq_x, uniq_y)

    return grid, smoothed


def plot_one_attack(attack_label, path):
    neg = load_z(path, NEGATIVE_SCHEME)

    if not neg:
        print(f"[warning] No vanilla negative scores found for {attack_label}. Skipping.")
        return

    fig, ax = plt.subplots()

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label="Chance"
    )

    COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    plotted = 0

    for idx, (scheme_key, scheme_display) in enumerate(METHODS):
        pos = load_z(path, scheme_key)

        if not pos:
            print(f"[warning] No scores found for {scheme_display} in {attack_label}. Skipping.")
            continue

        pts, auc = roc_points(pos, neg)
        gx, gy = smooth(pts)

        label = f"{scheme_display} — AUC {auc:.3f}"

        ax.plot(
            gx,
            gy,
            color=COLORS[idx % len(COLORS)],
            linewidth=2.6,
            label=label
        )

        plotted += 1

    if plotted == 0:
        print(f"[warning] No watermark methods plotted for {attack_label}.")
        plt.close(fig)
        return

    ax.set_title(f"ROC Curve on OPT-2.7B — {attack_label}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(loc="lower right", fontsize=9)

    out_name = f"roc_{safe_filename(attack_label)}.png"
    out_path = os.path.join(OUT_DIR, out_name)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved ROC for {attack_label} → {out_path}")


# ======================== MAIN ========================
def main():
    tasks = enabled_tasks()
    os.makedirs(OUT_DIR, exist_ok=True)

    for attack_label, path in tasks:
        plot_one_attack(attack_label, path)

    print("\nDone. Separate ROC plots saved in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()