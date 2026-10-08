# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict

import os
import json
import math
import re

from sklearn.metrics import roc_auc_score


# =========================================================
# TOGGLES
# =========================================================
ENABLE_ORIGINAL = True
ENABLE_PEGASUS = False
ENABLE_DIPPER = False
ENABLE_T5PARROT = False
ENABLE_ENDE_BACK = False
ENABLE_ENFR_BACK = False


# =========================================================
# FILE PATHS
# =========================================================
FILE_ORIGINAL = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "reviewer/outputs_c4_realnewslike_Llama-3-8B-Instruct/Recursive_Lightweight_Paraphrasing/Round2/"
    "detect_all.jsonl"
)

FILE_PEGASUS = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/"
    "Pegasus/new_check/detect_all.jsonl"
)

FILE_DIPPER = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "outputs_OpenGen_opt-2.7B_entropy_0.9/"
    "Dipper/detect_all.jsonl"
)

FILE_T5PARROT = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/"
    "detect_all.jsonl"
)

FILE_ENDE_BACK = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/"
    "English-German-English/detect_all.jsonl"
)

FILE_ENFR_BACK = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "outputs_c4_realnewslike_opt-2.7B_entropy_0.9/"
    "English-French-English/detect_all.jsonl"
)


# =========================================================
# AUTOMATIC OUTPUT SETTINGS
# =========================================================
# Example output filename:
#     auroc_results_Llama-3.1-8B.csv
#
# The CSV is saved automatically in the nearest parent
# generation folder whose name starts with "outputs_".
OUTPUT_FILE_PREFIX = "auroc_results"


MODEL_NAME_PATTERN = re.compile(
    r"(Llama-\d+(?:\.\d+)?-\d+(?:\.\d+)?B(?:-Instruct)?"
    r"|Qwen\d+(?:\.\d+)?-\d+(?:\.\d+)?B(?:-Instruct)?"
    r"|OPT-\d+(?:\.\d+)?B"
    r"|GPT-Neo-\d+(?:\.\d+)?B)",
    flags=re.IGNORECASE,
)


# =========================================================
# METHODS
# =========================================================
SCHEMES = OrderedDict([
    ("KGW", "kgw"),
    ("SWEET", "sweet"),
    ("Unigram", "unigram"),
    ("EWD", "ewd"),
    ("Rank-Aware", "rankaware"),
])

TARGET_FPR = 0.05


# =========================================================
# HELPERS
# =========================================================
def _fail(message: str):
    print("[ERROR]", message)
    raise SystemExit(1)


def _generation_directory(detection_path: str) -> str:
    """
    Return the generation folder associated with a detection file.

    For example:
        .../outputs_c4_realnewslike_Llama-3.1-8B/detect_all.jsonl

    returns:
        .../outputs_c4_realnewslike_Llama-3.1-8B

    This also works for attack files stored in subdirectories such as:
        .../outputs_.../Pegasus/new_check/detect_all.jsonl
    """
    current_directory = os.path.abspath(
        os.path.dirname(detection_path)
    )

    while True:
        directory_name = os.path.basename(current_directory)

        if directory_name.startswith("outputs_"):
            return current_directory

        parent_directory = os.path.dirname(current_directory)

        if parent_directory == current_directory:
            # Fallback: save beside detect_all.jsonl when no
            # parent directory starts with "outputs_".
            return os.path.abspath(
                os.path.dirname(detection_path)
            )

        current_directory = parent_directory


def _model_name(detection_path: str) -> str:
    """Extract the model name from the generation folder name."""
    generation_directory = _generation_directory(
        detection_path
    )

    directory_name = os.path.basename(
        generation_directory
    )

    match = MODEL_NAME_PATTERN.search(directory_name)

    if match:
        return match.group(1)

    # Safe fallback for an unfamiliar model naming format.
    safe_directory_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        directory_name,
    ).strip("_")

    return safe_directory_name or "unknown_model"


def _output_csv_path(detection_path: str) -> str:
    generation_directory = _generation_directory(
        detection_path
    )

    model_name = _model_name(detection_path)

    output_filename = (
        f"{OUTPUT_FILE_PREFIX}_{model_name}.csv"
    )

    return os.path.join(
        generation_directory,
        output_filename,
    )


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")

    return result if math.isfinite(result) else float("nan")


def _load_scores(path: str, scheme: str):
    scores = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line in file:
            try:
                record = json.loads(line)
            except Exception:
                continue

            if record.get("scheme") != scheme:
                continue

            value = record.get(
                "z",
                record.get("score", None),
            )

            score = _safe_float(value)

            if math.isfinite(score):
                scores.append(score)

    return scores


def _load_method_negative_scores(
    path: str,
    method_scheme: str,
):
    """
    New detection format:
        vanilla_kgw
        vanilla_sweet
        vanilla_unigram
        vanilla_ewd
        vanilla_rankaware

    The fallback to 'vanilla' supports older detection files.
    """

    method_negative_scheme = f"vanilla_{method_scheme}"

    negative_scores = _load_scores(
        path,
        method_negative_scheme,
    )

    if negative_scores:
        return negative_scores, method_negative_scheme

    fallback_scores = _load_scores(
        path,
        "vanilla",
    )

    if fallback_scores:
        print(
            f"[WARNING] No scores found for "
            f"'{method_negative_scheme}'. "
            f"Using old scheme 'vanilla'."
        )

        return fallback_scores, "vanilla"

    return [], method_negative_scheme


def _quantile(sorted_values, q: float):
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


def _tau_for_fpr(negative_scores, fpr: float):
    return _quantile(
        sorted(negative_scores),
        1.0 - fpr,
    )


def _safe_divide(numerator: float, denominator: float):
    return numerator / denominator if denominator else 0.0


def _counts_at_threshold(
    positive_scores,
    negative_scores,
    threshold: float,
):
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


def _auc(positive_scores, negative_scores):
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

    return roc_auc_score(
        labels,
        scores,
    )


def _fmt(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"

    return f"{value:.3f}"


# =========================================================
# COMPUTE METRICS
# =========================================================
def compute_for_file(path: str):
    results = OrderedDict()

    for method_name, method_scheme in SCHEMES.items():

        # -------------------------------------------------
        # Each detector's own vanilla scores
        # -------------------------------------------------
        negative_scores, negative_scheme = (
            _load_method_negative_scores(
                path,
                method_scheme,
            )
        )

        if not negative_scores:
            print(
                f"[WARNING] No negative scores found for "
                f"method '{method_scheme}' using scheme "
                f"'{negative_scheme}' in:\n{path}"
            )

        # -------------------------------------------------
        # Corresponding watermarked scores
        # -------------------------------------------------
        positive_scores = _load_scores(
            path,
            method_scheme,
        )

        if not positive_scores:
            print(
                f"[WARNING] No positive scores found for "
                f"scheme '{method_scheme}' in:\n{path}"
            )

        if not negative_scores:
            results[method_name] = {
                "tpr": float("nan"),
                "fpr": float("nan"),
                "precision": float("nan"),
                "f1": float("nan"),
                "auroc": float("nan"),
                "tau": float("nan"),
                "tp": 0,
                "fp": 0,
                "fn": len(positive_scores),
                "tn": 0,
                "n_pos": len(positive_scores),
                "n_neg": 0,
                "negative_scheme": negative_scheme,
            }

            continue

        # -------------------------------------------------
        # Method-specific 5% FPR threshold
        # -------------------------------------------------
        threshold = _tau_for_fpr(
            negative_scores,
            TARGET_FPR,
        )

        (
            true_positive,
            false_positive,
            false_negative,
            true_negative,
        ) = _counts_at_threshold(
            positive_scores,
            negative_scores,
            threshold,
        )

        tpr = _safe_divide(
            true_positive,
            true_positive + false_negative,
        )

        fpr = _safe_divide(
            false_positive,
            false_positive + true_negative,
        )

        precision = _safe_divide(
            true_positive,
            true_positive + false_positive,
        )

        f1 = _safe_divide(
            2.0 * precision * tpr,
            precision + tpr,
        )

        auroc = _auc(
            positive_scores,
            negative_scores,
        )

        results[method_name] = {
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "f1": f1,
            "auroc": auroc,
            "tau": threshold,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "tn": true_negative,
            "n_pos": len(positive_scores),
            "n_neg": len(negative_scores),
            "negative_scheme": negative_scheme,
        }

        print(
            f"{method_name:<12} | "
            f"negative={negative_scheme:<22} | "
            f"n_neg={len(negative_scores):<5} | "
            f"n_pos={len(positive_scores):<5} | "
            f"tau={threshold:.4f} | "
            f"actual_fpr={fpr:.4f}"
        )

    return results


def _write_results_csv(
    output_path: str,
    grouped_results,
):
    """Write all results belonging to one generation folder."""
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "attack,"
            "method,"
            "negative_scheme,"
            "tpr_at_5_fpr,"
            "actual_fpr,"
            "precision,"
            "f1_at_5,"
            "auroc,"
            "tau,"
            "tp,"
            "fp,"
            "fn,"
            "tn,"
            "n_pos,"
            "n_neg\n"
        )

        for attack_name, attack_results in grouped_results.items():
            for method_name, metrics in attack_results.items():
                file.write(
                    f"{attack_name},"
                    f"{method_name},"
                    f"{metrics['negative_scheme']},"
                    f"{metrics['tpr']:.6f},"
                    f"{metrics['fpr']:.6f},"
                    f"{metrics['precision']:.6f},"
                    f"{metrics['f1']:.6f},"
                    f"{metrics['auroc']:.6f},"
                    f"{metrics['tau']:.6f},"
                    f"{metrics['tp']},"
                    f"{metrics['fp']},"
                    f"{metrics['fn']},"
                    f"{metrics['tn']},"
                    f"{metrics['n_pos']},"
                    f"{metrics['n_neg']}\n"
                )


# =========================================================
# MAIN
# =========================================================
def main():
    datasets = OrderedDict()

    if ENABLE_ORIGINAL:
        datasets["No Attack"] = FILE_ORIGINAL

    if ENABLE_PEGASUS:
        datasets["Pegasus"] = FILE_PEGASUS

    if ENABLE_DIPPER:
        datasets["DIPPER"] = FILE_DIPPER

    if ENABLE_T5PARROT:
        datasets["T5-Parrot"] = FILE_T5PARROT

    if ENABLE_ENDE_BACK:
        datasets["EN-DE-EN"] = FILE_ENDE_BACK

    if ENABLE_ENFR_BACK:
        datasets["EN-FR-EN"] = FILE_ENFR_BACK

    if not datasets:
        _fail("No dataset is enabled.")

    all_results = OrderedDict()

    # One output group per generation directory. This allows
    # enabled files from different models or experiments to be
    # saved automatically in their own respective folders.
    output_groups = OrderedDict()

    for dataset_name, path in datasets.items():
        if not os.path.exists(path):
            _fail(f"Missing detection file: {path}")

        print("\n" + "=" * 80)
        print("Loading:", dataset_name)
        print("File:", path)
        print("=" * 80)

        dataset_results = compute_for_file(path)
        all_results[dataset_name] = dataset_results

        output_csv = _output_csv_path(path)

        if output_csv not in output_groups:
            output_groups[output_csv] = OrderedDict()

        output_groups[output_csv][
            dataset_name
        ] = dataset_results

    # =====================================================
    # TABLE 1
    # =====================================================
    print(
        "\n"
        "================ TABLE 1: "
        "TPR@5%-FPR, F1@5%, AUROC ================\n"
    )

    header = f"{'Method':<16}"

    for dataset_name in all_results:
        header += f"| {dataset_name:^30} "

    print(header)

    subheader = f"{'':<16}"

    for _ in all_results:
        subheader += (
            "| TPR@5%-FPR   F1@5%    AUROC "
        )

    print(subheader)
    print("-" * len(header))

    for method_name in SCHEMES:
        row = f"{method_name:<16}"

        for dataset_name in all_results:
            metrics = all_results[
                dataset_name
            ][method_name]

            row += (
                f"| {_fmt(metrics['tpr']):>10}   "
                f"{_fmt(metrics['f1']):>7}   "
                f"{_fmt(metrics['auroc']):>7} "
            )

        print(row)

    print("\n" + "=" * 80)

    # =====================================================
    # TABLE 2: AUROC ONLY
    # =====================================================
    print(
        "\n"
        "======================== "
        "TABLE 2: AUROC "
        "========================\n"
    )

    header = f"{'Method':<16}"

    for dataset_name in all_results:
        header += f"| {dataset_name:^10} "

    print(header)
    print("-" * len(header))

    for method_name in SCHEMES:
        row = f"{method_name:<16}"

        for dataset_name in all_results:
            auroc = all_results[
                dataset_name
            ][method_name]["auroc"]

            row += f"| {_fmt(auroc):>8} "

        print(row)

    print("\n" + "=" * 80)

    # =====================================================
    # AUTOMATIC CSV OUTPUT
    # =====================================================
    for output_csv, grouped_results in output_groups.items():
        _write_results_csv(
            output_csv,
            grouped_results,
        )

        print("\nCSV saved ->", output_csv)


if __name__ == "__main__":
    main()