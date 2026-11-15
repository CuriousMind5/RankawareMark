# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import csv
import json
import math

from sklearn.metrics import roc_auc_score


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main"
)

INPUT_DIR = BASE_DIR / "reviewer/Rankware_entropy"

DETECTION_FILES = OrderedDict([
    (
        "OPT-2.7B",
        INPUT_DIR
        / "detect_all_opt-2.7b_entropy_ablation.jsonl",
    ),
    (
        "OPT-6.7B",
        INPUT_DIR
        / "detect_all_opt-6.7b_entropy_ablation.jsonl",
    ),
])

OUT_CSV = (
    INPUT_DIR
    / "rankaware_entropy_ablation_metrics.csv"
)


# =========================================================
# ABLATION VARIANTS
# =========================================================
VARIANTS = OrderedDict([
    (
        "Rank-Aware with entropy",
        {
            "entropy_tau": 0.9,
            "positive_scheme": (
                "rankaware_with_entropy_tau0p9"
            ),
            "negative_scheme": (
                "vanilla_with_entropy_tau0p9"
            ),
        },
    ),
    (
        "Rank-Aware without entropy",
        {
            "entropy_tau": 0.0,
            "positive_scheme": (
                "rankaware_without_entropy_tau0"
            ),
            "negative_scheme": (
                "vanilla_without_entropy_tau0"
            ),
        },
    ),
])


# Target operating point.
TARGET_FPR = 0.05


# =========================================================
# HELPERS
# =========================================================
def fail(message: str) -> None:
    print("[ERROR]", message)
    raise SystemExit(1)


def safe_float(value) -> float:
    try:
        result = float(value)

    except (TypeError, ValueError):
        return float("nan")

    if math.isfinite(result):
        return result

    return float("nan")


def load_scores(
    path: Path,
    scheme: str,
) -> list[float]:
    scores = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                print(
                    f"[WARNING] Invalid JSON at "
                    f"{path}:{line_number}"
                )
                continue

            if record.get("scheme") != scheme:
                continue

            value = record.get(
                "z",
                record.get("score"),
            )

            score = safe_float(value)

            if math.isfinite(score):
                scores.append(score)

    return scores


def quantile(
    sorted_values: list[float],
    q: float,
) -> float:
    count = len(sorted_values)

    if count == 0:
        return float("nan")

    position = q * (count - 1)

    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return sorted_values[lower]

    fraction = position - lower

    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def threshold_for_fpr(
    negative_scores: list[float],
    target_fpr: float,
) -> float:
    return quantile(
        sorted(negative_scores),
        1.0 - target_fpr,
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def confusion_counts(
    positive_scores: list[float],
    negative_scores: list[float],
    threshold: float,
) -> tuple[int, int, int, int]:
    true_positive = sum(
        score >= threshold
        for score in positive_scores
    )

    false_negative = (
        len(positive_scores)
        - true_positive
    )

    false_positive = sum(
        score >= threshold
        for score in negative_scores
    )

    true_negative = (
        len(negative_scores)
        - false_positive
    )

    return (
        true_positive,
        false_positive,
        false_negative,
        true_negative,
    )


def calculate_auroc(
    positive_scores: list[float],
    negative_scores: list[float],
) -> float:
    if not positive_scores or not negative_scores:
        return float("nan")

    labels = (
        [1] * len(positive_scores)
        + [0] * len(negative_scores)
    )

    scores = (
        positive_scores
        + negative_scores
    )

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


def format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "nan"

    return f"{value:.3f}"


# =========================================================
# CALCULATE ONE VARIANT
# =========================================================
def calculate_variant_metrics(
    detection_file: Path,
    positive_scheme: str,
    negative_scheme: str,
) -> dict:
    positive_scores = load_scores(
        path=detection_file,
        scheme=positive_scheme,
    )

    negative_scores = load_scores(
        path=detection_file,
        scheme=negative_scheme,
    )

    if not positive_scores:
        print(
            f"[WARNING] No positive scores for "
            f"'{positive_scheme}' in {detection_file}"
        )

    if not negative_scores:
        print(
            f"[WARNING] No negative scores for "
            f"'{negative_scheme}' in {detection_file}"
        )

    if not positive_scores or not negative_scores:
        return {
            "tpr": float("nan"),
            "fpr": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "threshold": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": len(positive_scores),
            "tn": len(negative_scores),
            "n_pos": len(positive_scores),
            "n_neg": len(negative_scores),
        }

    threshold = threshold_for_fpr(
        negative_scores=negative_scores,
        target_fpr=TARGET_FPR,
    )

    (
        true_positive,
        false_positive,
        false_negative,
        true_negative,
    ) = confusion_counts(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        threshold=threshold,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    actual_fpr = safe_divide(
        false_positive,
        false_positive + true_negative,
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    auroc = calculate_auroc(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
    )

    return {
        "tpr": recall,
        "fpr": actual_fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "threshold": threshold,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "n_pos": len(positive_scores),
        "n_neg": len(negative_scores),
    }


# =========================================================
# PROCESS ONE MODEL
# =========================================================
def calculate_model_metrics(
    model_name: str,
    detection_file: Path,
) -> OrderedDict:
    if not detection_file.exists():
        fail(
            f"Missing detection file for {model_name}: "
            f"{detection_file}"
        )

    print("\n" + "=" * 80)
    print("Model:", model_name)
    print("Detection file:", detection_file)
    print("=" * 80)

    model_results = OrderedDict()

    for variant_name, variant in VARIANTS.items():
        metrics = calculate_variant_metrics(
            detection_file=detection_file,
            positive_scheme=variant[
                "positive_scheme"
            ],
            negative_scheme=variant[
                "negative_scheme"
            ],
        )

        model_results[variant_name] = metrics

        print(
            f"{variant_name:<32} | "
            f"tau_H={variant['entropy_tau']:<4} | "
            f"n_pos={metrics['n_pos']:<5} | "
            f"n_neg={metrics['n_neg']:<5} | "
            f"threshold={metrics['threshold']:.4f} | "
            f"TPR={metrics['tpr']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"AUROC={metrics['auroc']:.4f}"
        )

    return model_results


# =========================================================
# PRINT COMPARISON TABLE
# =========================================================
def print_comparison_table(
    all_results: OrderedDict,
) -> None:
    print(
        "\n"
        "================ RANK-AWARE ENTROPY ABLATION "
        "================\n"
    )

    header = f"{'Variant':<32}"

    for model_name in all_results:
        header += f"| {model_name:^30} "

    print(header)

    subheader = f"{'':<32}"

    for _ in all_results:
        subheader += (
            "| TPR@5%     F1       AUROC     "
        )

    print(subheader)
    print("-" * len(header))

    for variant_name in VARIANTS:
        row = f"{variant_name:<32}"

        for model_name in all_results:
            metrics = all_results[
                model_name
            ][variant_name]

            row += (
                f"| "
                f"{format_metric(metrics['tpr']):>7}   "
                f"{format_metric(metrics['f1']):>7}   "
                f"{format_metric(metrics['auroc']):>7}   "
            )

        print(row)

    print("\n" + "=" * 100)


# =========================================================
# SAVE CSV
# =========================================================
def save_csv(
    all_results: OrderedDict,
) -> None:
    with open(
        OUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "variant",
            "entropy_tau",
            "positive_scheme",
            "negative_scheme",
            "tpr_at_5_fpr",
            "actual_fpr",
            "precision",
            "recall",
            "f1_at_5_fpr",
            "auroc",
            "threshold_at_5_fpr",
            "tp",
            "fp",
            "fn",
            "tn",
            "n_pos",
            "n_neg",
        ])

        for model_name, model_results in (
            all_results.items()
        ):
            for variant_name, metrics in (
                model_results.items()
            ):
                variant = VARIANTS[variant_name]

                writer.writerow([
                    model_name,
                    variant_name,
                    variant["entropy_tau"],
                    variant["positive_scheme"],
                    variant["negative_scheme"],
                    f"{metrics['tpr']:.6f}",
                    f"{metrics['fpr']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['auroc']:.6f}",
                    f"{metrics['threshold']:.6f}",
                    metrics["tp"],
                    metrics["fp"],
                    metrics["fn"],
                    metrics["tn"],
                    metrics["n_pos"],
                    metrics["n_neg"],
                ])

    print("CSV saved:", OUT_CSV)


# =========================================================
# MAIN
# =========================================================
def main():
    all_results = OrderedDict()

    for model_name, detection_file in (
        DETECTION_FILES.items()
    ):
        all_results[model_name] = (
            calculate_model_metrics(
                model_name=model_name,
                detection_file=detection_file,
            )
        )

    print_comparison_table(all_results)
    save_csv(all_results)


if __name__ == "__main__":
    main()
