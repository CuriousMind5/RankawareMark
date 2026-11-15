# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# Must be set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gc
import json
import math
import statistics as st
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.transformers_config import TransformersConfig
from watermark.rankaware import RankAware, RankAwareConfig


# =========================================================
# CLI
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--z_threshold", type=float, default=4.0)

args = parser.parse_args()


# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

DTYPE = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

print("Running on device:", DEVICE)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main"
)

INPUT_DIR = BASE_DIR / "reviewer/Rankware_entropy"

RANKAWARE_CONFIG_PATH = (
    BASE_DIR / "config/RankAware.json"
)

INPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# MODELS
# =========================================================
MODELS = [
    {
        "name": "opt-2.7b",
        "path": Path(
            "/home/sy/watermark_unlearning_experiment/"
            "models/facebook/opt-2.7b/"
        ),
    },
    {
        "name": "opt-6.7b",
        "path": Path(
            "/home/sy/watermark_unlearning_experiment/"
            "models/facebook/opt-6.7B/"
        ),
    },
]


# =========================================================
# RANK-AWARE VARIANTS
# =========================================================
VARIANTS = [
    {
        "name": "with_entropy_tau0p9",
        "detector_name": "rankaware_tau0p9",
        "entropy_tau": 0.9,
        "entropy_gate_enabled": True,
    },
    {
        "name": "without_entropy_tau0",
        "detector_name": "rankaware_tau0",
        "entropy_tau": 0.0,
        "entropy_gate_enabled": False,
    },
]


# =========================================================
# HELPERS
# =========================================================
def load_json(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            f"Expected a JSON list in {path}, "
            f"but found {type(data)}"
        )

    return data


def extract_text(record: dict) -> str:
    for key in [
        "sampled",
        "text",
        "completion",
        "output",
        "generated_text",
    ]:
        if key in record:
            return str(record[key]).strip()

    raise KeyError(
        "No valid generated-text field found. Expected one of: "
        "sampled, text, completion, output, generated_text."
    )


def token_len(text: str, tokenizer) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


def extract_score(output: dict) -> float:
    candidate_keys = [
        "score",
        "z_score",
        "z",
        "confidence",
        "detection_score",
        "watermark_score",
    ]

    for key in candidate_keys:
        if key not in output:
            continue

        try:
            value = float(output[key])

            if math.isfinite(value):
                return value

        except (TypeError, ValueError):
            continue

    return float("nan")


def detect_one(
    text: str,
    detector,
) -> tuple[float, str | None]:
    try:
        with torch.inference_mode():
            output = detector.detect_watermark(
                text,
                return_dict=True,
            )

        if not isinstance(output, dict):
            raise TypeError(
                "Detector output must be a dictionary, "
                f"but got {type(output)}"
            )

        score = extract_score(output)

        if not math.isfinite(score):
            return (
                float("nan"),
                f"No finite detection score found. "
                f"Available keys: {list(output.keys())}",
            )

        return score, None

    except Exception as error:
        return float("nan"), str(error)


def vanilla_file_path(model_name: str) -> Path:
    return INPUT_DIR / (
        f"vanilla_{model_name}_"
        f"n{args.num_samples}_"
        f"tokens{args.max_new_tokens}.json"
    )


def rankaware_file_path(
    model_name: str,
    variant_name: str,
) -> Path:
    return INPUT_DIR / (
        f"rankaware_{model_name}_{variant_name}_"
        f"n{args.num_samples}_"
        f"tokens{args.max_new_tokens}.json"
    )


def validate_matching_data(
    vanilla_data: list,
    watermarked_data: list,
    vanilla_path: Path,
    watermarked_path: Path,
) -> None:
    if len(vanilla_data) != len(watermarked_data):
        raise ValueError(
            "Vanilla and watermarked sample counts differ:\n"
            f"Vanilla: {len(vanilla_data)} in {vanilla_path}\n"
            f"Watermarked: {len(watermarked_data)} "
            f"in {watermarked_path}"
        )

    prompt_mismatches = 0

    for vanilla_record, watermark_record in zip(
        vanilla_data,
        watermarked_data,
    ):
        vanilla_prompt = vanilla_record.get("original")
        watermark_prompt = watermark_record.get("original")

        if (
            vanilla_prompt is not None
            and watermark_prompt is not None
            and vanilla_prompt != watermark_prompt
        ):
            prompt_mismatches += 1

    if prompt_mismatches > 0:
        raise ValueError(
            f"Found {prompt_mismatches} prompt mismatches "
            f"between {vanilla_path.name} and "
            f"{watermarked_path.name}."
        )

    print(
        "Validated matching samples and prompts:",
        len(vanilla_data),
    )


# =========================================================
# LOAD MODEL
# =========================================================
def load_model_and_tokenizer(
    model_name: str,
    model_path: Path,
):
    print("\n" + "#" * 75)
    print("Loading model:", model_name)
    print("Model path:", model_path)
    print("#" * 75)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=DTYPE,
    ).to(DEVICE)

    model.eval()

    model.generation_config.pad_token_id = (
        tokenizer.eos_token_id
    )

    return model, tokenizer


# =========================================================
# BUILD MATCHING RANK-AWARE DETECTOR
# =========================================================
def build_rankaware_detector(
    tf_cfg,
    entropy_tau: float,
):
    """
    Construct the Rank-Aware detector using the same entropy
    threshold used during generation.
    """
    config = RankAwareConfig(
        RANKAWARE_CONFIG_PATH,
        tf_cfg,
    )

    if not hasattr(config, "entropy_tau"):
        available_attributes = [
            name
            for name in dir(config)
            if not name.startswith("_")
        ]

        raise AttributeError(
            "RankAwareConfig has no 'entropy_tau' attribute.\n"
            f"Available attributes: {available_attributes}"
        )

    original_tau = config.entropy_tau
    config.entropy_tau = float(entropy_tau)

    if isinstance(config.config_dict, dict):
        config.config_dict["entropy_tau"] = float(
            entropy_tau
        )

    print(
        "Detector entropy tau override:",
        f"{original_tau} -> {config.entropy_tau}",
    )

    detector = RankAware(
        config,
        tf_cfg,
    )

    return config, detector


# =========================================================
# PROCESS ONE FILE
# =========================================================
def process_dataset(
    data: list,
    detector,
    model_name: str,
    detector_name: str,
    variant_name: str,
    entropy_tau: float,
    source_scheme: str,
    output_scheme: str,
    label: int,
    tokenizer,
    results: list,
):
    description = (
        f"{model_name}: {detector_name} -> {source_scheme}"
    )

    for index, record in enumerate(
        tqdm(data, desc=description)
    ):
        base_result = {
            "model": model_name,
            "variant": variant_name,
            "entropy_tau": entropy_tau,
            "scheme": output_scheme,
            "detector_scheme": detector_name,
            "source_scheme": source_scheme,
            "idx": record.get("id", index),
            "label": label,
        }

        try:
            text = extract_text(record)

        except Exception as error:
            results.append(
                {
                    **base_result,
                    "z": float("nan"),
                    "score": float("nan"),
                    "flagged": False,
                    "error": str(error),
                }
            )
            continue

        if token_len(text, tokenizer) < 2:
            results.append(
                {
                    **base_result,
                    "z": float("nan"),
                    "score": float("nan"),
                    "flagged": False,
                    "error": (
                        "Text contains fewer than two tokens."
                    ),
                }
            )
            continue

        score, error = detect_one(
            text=text,
            detector=detector,
        )

        flagged = bool(
            math.isfinite(score)
            and score >= args.z_threshold
        )

        result = {
            **base_result,
            "z": score,
            "score": score,
            "flagged": flagged,
        }

        if error is not None:
            result["error"] = error

        results.append(result)


# =========================================================
# SUMMARY
# =========================================================
def build_summary(results: list) -> dict:
    statistics = {}

    for result in results:
        scheme = result["scheme"]

        if scheme not in statistics:
            statistics[scheme] = {
                "model": result["model"],
                "variant": result["variant"],
                "entropy_tau": result["entropy_tau"],
                "detector_scheme": (
                    result["detector_scheme"]
                ),
                "source_scheme": (
                    result["source_scheme"]
                ),
                "label": result["label"],
                "total": 0,
                "valid": 0,
                "skipped": 0,
                "flagged": 0,
                "scores": [],
            }

        values = statistics[scheme]
        values["total"] += 1

        score = result["score"]

        if math.isfinite(score):
            values["valid"] += 1
            values["scores"].append(score)

            if result["flagged"]:
                values["flagged"] += 1

        else:
            values["skipped"] += 1

    return statistics


def save_results(
    results: list,
    statistics: dict,
    out_jsonl: Path,
    out_csv: Path,
) -> None:
    with open(out_jsonl, "w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(out_csv, "w", encoding="utf-8") as file:
        file.write(
            "model,"
            "variant,"
            "entropy_tau,"
            "scheme,"
            "detector_scheme,"
            "source_scheme,"
            "label,"
            "total,"
            "valid,"
            "skipped,"
            "mean_z,"
            "std_z,"
            "percent_flagged\n"
        )

        for scheme, values in statistics.items():
            scores = values["scores"]

            mean_z = (
                st.mean(scores)
                if scores
                else float("nan")
            )

            std_z = (
                st.pstdev(scores)
                if len(scores) > 1
                else 0.0
            )

            percent_flagged = (
                100.0
                * values["flagged"]
                / values["valid"]
                if values["valid"] > 0
                else 0.0
            )

            file.write(
                f"{values['model']},"
                f"{values['variant']},"
                f"{values['entropy_tau']},"
                f"{scheme},"
                f"{values['detector_scheme']},"
                f"{values['source_scheme']},"
                f"{values['label']},"
                f"{values['total']},"
                f"{values['valid']},"
                f"{values['skipped']},"
                f"{mean_z:.6f},"
                f"{std_z:.6f},"
                f"{percent_flagged:.2f}\n"
            )


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 75)
    print("RANK-AWARE ENTROPY ABLATION DETECTION")
    print("=" * 75)
    print("Input directory:", INPUT_DIR)
    print("Device:", DEVICE)
    print("Z threshold:", args.z_threshold)

    for model_info in MODELS:
        model_name = model_info["name"]
        model_path = model_info["path"]

        vanilla_path = vanilla_file_path(model_name)
        vanilla_data = load_json(vanilla_path)

        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            model_path=model_path,
        )

        model_device = next(model.parameters()).device

        vocab_size = (
            model.get_output_embeddings()
            .weight.shape[0]
        )

        print("Model device:", model_device)
        print("Vocabulary size:", vocab_size)

        tf_cfg = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            device=model_device,
            vocab_size=vocab_size,
        )

        model_results = []

        for variant in VARIANTS:
            variant_name = variant["name"]
            detector_name = variant["detector_name"]
            entropy_tau = variant["entropy_tau"]

            watermarked_path = rankaware_file_path(
                model_name=model_name,
                variant_name=variant_name,
            )

            watermarked_data = load_json(
                watermarked_path
            )

            validate_matching_data(
                vanilla_data=vanilla_data,
                watermarked_data=watermarked_data,
                vanilla_path=vanilla_path,
                watermarked_path=watermarked_path,
            )

            print("\n" + "=" * 75)
            print("Model:", model_name)
            print("Detector:", detector_name)
            print("Entropy tau:", entropy_tau)
            print("=" * 75)

            detector_config, detector = (
                build_rankaware_detector(
                    tf_cfg=tf_cfg,
                    entropy_tau=entropy_tau,
                )
            )

            # Score vanilla using the matching detector.
            process_dataset(
                data=vanilla_data,
                detector=detector,
                model_name=model_name,
                detector_name=detector_name,
                variant_name=variant_name,
                entropy_tau=entropy_tau,
                source_scheme="vanilla",
                output_scheme=f"vanilla_{variant_name}",
                label=0,
                tokenizer=tokenizer,
                results=model_results,
            )

            # Score watermarked text using the same detector.
            process_dataset(
                data=watermarked_data,
                detector=detector,
                model_name=model_name,
                detector_name=detector_name,
                variant_name=variant_name,
                entropy_tau=entropy_tau,
                source_scheme="rankaware",
                output_scheme=f"rankaware_{variant_name}",
                label=1,
                tokenizer=tokenizer,
                results=model_results,
            )

            del detector
            del detector_config

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        statistics = build_summary(model_results)

        out_jsonl = INPUT_DIR / (
            f"detect_all_{model_name}_entropy_ablation.jsonl"
        )

        out_csv = INPUT_DIR / (
            f"detect_summary_{model_name}_entropy_ablation.csv"
        )

        save_results(
            results=model_results,
            statistics=statistics,
            out_jsonl=out_jsonl,
            out_csv=out_csv,
        )

        print("\n" + "=" * 75)
        print("DETECTION SUMMARY:", model_name)
        print("=" * 75)

        for scheme, values in statistics.items():
            scores = values["scores"]

            mean_z = (
                st.mean(scores)
                if scores
                else float("nan")
            )

            percent_flagged = (
                100.0
                * values["flagged"]
                / values["valid"]
                if values["valid"] > 0
                else 0.0
            )

            print(
                f"{scheme:<38} "
                f"valid={values['valid']:<5} "
                f"skipped={values['skipped']:<5} "
                f"mean_z={mean_z:>9.4f} "
                f"flagged={percent_flagged:>7.2f}%"
            )

        print("Details:", out_jsonl)
        print("Summary:", out_csv)

        del tf_cfg
        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll entropy-ablation detection completed.")


if __name__ == "__main__":
    main()
