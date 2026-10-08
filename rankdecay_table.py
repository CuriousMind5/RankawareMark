# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score

BASE_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_c4_realnewslike_opt-6.7_entropy_0.9/rank_decay"

RANKS = [20, 40, 60, 80, 100, 120, 140, 160, 180]

rows = []

for r in RANKS:

    folder = f"vanilla_vs_rank{r}"
    path = os.path.join(BASE_DIR, folder, "detect_all.jsonl")

    if not os.path.exists(path):
        print(f"Missing rank {r}")
        continue

    labels = []
    scores = []
    preds = []

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)

            labels.append(obj["label"])
            scores.append(obj["z"])
            preds.append(1 if obj["flagged"] else 0)

    labels = np.array(labels)
    scores = np.array(scores)
    preds  = np.array(preds)

    # ---- Metrics ----
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    TPR = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * precision * TPR / (precision + TPR)

    AUC = roc_auc_score(labels, scores)

    rows.append((r, TPR*100, F1, AUC))

# ================= PRINT ACL LATEX =================

print("\n================ LATEX TABLE ================\n")

print(r"\begin{table}[t]")
print(r"\centering")
print(r"\footnotesize")
print(r"\begin{tabular}{c c c c}")
print(r"\toprule")
print(r"Rank Decay & TPR@5\% $\uparrow$ & F1@5\% $\uparrow$ & AUROC $\uparrow$ \\")
print(r"\midrule")

for r, tpr, f1, auc in rows:
    print(f"{r} & {tpr:.2f} & {f1:.3f} & {auc:.3f} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Effect of rank decay on Rank-Aware watermark detection under clean generation (No Attack).}")
print(r"\label{tab:rank_decay_noattack}")
print(r"\end{table}")
