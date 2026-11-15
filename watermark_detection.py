# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# ---------------- OFFLINE ENV ----------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
import statistics as st
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.transformers_config import TransformersConfig

# ---------------- WATERMARKS ----------------
from watermark.kgw.kgw import KGW, KGWConfig
from watermark.sweet.sweet import SWEET, SWEETConfig
from watermark.ewd.ewd import EWD, EWDConfig
from watermark.unigram.unigram import Unigram, UnigramConfig
from watermark.rankaware import RankAware, RankAwareConfig


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main"
)

MODEL_PATH = Path(
    "/home/sy/watermark_unlearning_experiment/models/Llama-3-8B-instruct"
)

INPUT_DIR = BASE_DIR / (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "reviewer/outputs_c4_realnewslike_Llama-3-8B-Instruct/Recursive_Lightweight_Paraphrasing/Round2/"
)

VANILLA_FILE = INPUT_DIR / "vanilla_n500.json"

METHOD_FILES = {
    "kgw": "kgw_n500.json",
    "unigram": "unigram_n500.json",
    "sweet": "sweet_n500.json",
    "ewd": "ewd_n500.json",
    "rankaware": "rankaware_n500.json",
}

OUT_DIR = INPUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "detect_all.jsonl"
OUT_CSV = OUT_DIR / "detect_summary.csv"

Z_THRESHOLD = 4.0

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

DTYPE = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


# =========================================================
# HELPERS
# =========================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        "No valid text field found. Expected sampled, text, "
        "completion, output, or generated_text."
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
                f"Detector output must be a dictionary, "
                f"got {type(output)}"
            )

        return extract_score(output), None

    except Exception as error:
        return float("nan"), str(error)


# =========================================================
# LOAD MODEL
# =========================================================
def load_model_and_tokenizer():
    print(f">>> Loading model: {MODEL_PATH.name}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=DTYPE,
        device_map=(
            "auto"
            if torch.cuda.is_available()
            else None
        ),
    )

    model.eval()

    if not torch.cuda.is_available():
        model = model.to(DEVICE)

    return model, tokenizer


# =========================================================
# BUILD DETECTORS
# =========================================================
def build_detectors(tf_cfg):
    return {
        "kgw": KGW(
            KGWConfig(
                BASE_DIR / "config/KGW.json",
                tf_cfg,
            ),
            tf_cfg,
        ),

        "unigram": Unigram(
            UnigramConfig(
                BASE_DIR / "config/Unigram.json",
                tf_cfg,
            ),
            tf_cfg,
        ),

        "sweet": SWEET(
            SWEETConfig(
                BASE_DIR / "config/SWEET.json",
                tf_cfg,
            ),
            tf_cfg,
        ),

        "ewd": EWD(
            EWDConfig(
                BASE_DIR / "config/EWD.json",
                tf_cfg,
            ),
            tf_cfg,
        ),

        "rankaware": RankAware(
            RankAwareConfig(
                BASE_DIR / "config/RankAware.json",
                tf_cfg,
            ),
            tf_cfg,
        ),
    }


# =========================================================
# PROCESS ONE DATASET
# =========================================================
def process_dataset(
    data,
    detector,
    detector_name: str,
    source_scheme: str,
    output_scheme: str,
    label: int,
    tokenizer,
    results: list,
):
    description = (
        f"{detector_name} detector -> {source_scheme}"
    )

    for idx, record in enumerate(
        tqdm(data, desc=description)
    ):
        try:
            text = extract_text(record)

        except Exception as error:
            results.append({
                "scheme": output_scheme,
                "detector_scheme": detector_name,
                "source_scheme": source_scheme,
                "idx": idx,
                "z": float("nan"),
                "score": float("nan"),
                "flagged": False,
                "label": label,
                "error": str(error),
            })
            continue

        if token_len(text, tokenizer) < 2:
            results.append({
                "scheme": output_scheme,
                "detector_scheme": detector_name,
                "source_scheme": source_scheme,
                "idx": idx,
                "z": float("nan"),
                "score": float("nan"),
                "flagged": False,
                "label": label,
                "error": (
                    "Text contains fewer than two tokens."
                ),
            })
            continue

        score, error = detect_one(
            text=text,
            detector=detector,
        )

        flagged = bool(
            math.isfinite(score)
            and score >= Z_THRESHOLD
        )

        result = {
            "scheme": output_scheme,
            "detector_scheme": detector_name,
            "source_scheme": source_scheme,
            "idx": idx,
            "z": score,
            "score": score,
            "flagged": flagged,
            "label": label,
        }

        if error is not None:
            result["error"] = error

        results.append(result)


# =========================================================
# SUMMARY
# =========================================================
def build_summary(results):
    stats = {}

    for result in results:
        scheme = result["scheme"]

        if scheme not in stats:
            stats[scheme] = {
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

        stats[scheme]["total"] += 1

        score = result["score"]

        if math.isfinite(score):
            stats[scheme]["valid"] += 1
            stats[scheme]["scores"].append(score)

            if result["flagged"]:
                stats[scheme]["flagged"] += 1

        else:
            stats[scheme]["skipped"] += 1

    return stats


def save_results(results, stats):
    with open(
        OUT_JSONL,
        "w",
        encoding="utf-8",
    ) as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(
        OUT_CSV,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
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

        for scheme, values in stats.items():
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

            f.write(
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
    print("=" * 70)
    print("METHOD-SPECIFIC WATERMARK DETECTION")
    print("=" * 70)
    print("Input directory:", INPUT_DIR)
    print("Device:", DEVICE)

    vanilla_data = load_json(VANILLA_FILE)

    model, tokenizer = load_model_and_tokenizer()

    model_device = next(model.parameters()).device

    vocab_size = (
        model.get_output_embeddings()
        .weight.shape[0]
    )

    print("Model device:", model_device)
    print("Model vocabulary size:", vocab_size)

    tf_cfg = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=model_device,
        vocab_size=vocab_size,
    )

    print("\n>>> Building detectors...")

    detectors = build_detectors(tf_cfg)

    results = []

    for method_name, filename in (
        METHOD_FILES.items()
    ):
        method_path = INPUT_DIR / filename

        if not method_path.exists():
            print(
                f"[SKIP] {method_name}: "
                f"file not found: {method_path}"
            )
            continue

        print("\n" + "=" * 70)
        print(
            f"Running {method_name.upper()} detector"
        )
        print("=" * 70)

        watermarked_data = load_json(
            method_path
        )

        detector = detectors[method_name]

        # -------------------------------------------------
        # Same detector on vanilla text
        # -------------------------------------------------
        process_dataset(
            data=vanilla_data,
            detector=detector,
            detector_name=method_name,
            source_scheme="vanilla",
            output_scheme=f"vanilla_{method_name}",
            label=0,
            tokenizer=tokenizer,
            results=results,
        )

        # -------------------------------------------------
        # Same detector on its own watermarked text
        # -------------------------------------------------
        process_dataset(
            data=watermarked_data,
            detector=detector,
            detector_name=method_name,
            source_scheme=method_name,
            output_scheme=method_name,
            label=1,
            tokenizer=tokenizer,
            results=results,
        )

    stats = build_summary(results)

    save_results(
        results=results,
        stats=stats,
    )

    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    for scheme, values in stats.items():
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
            f"{scheme:<22} "
            f"detector="
            f"{values['detector_scheme']:<10} "
            f"source="
            f"{values['source_scheme']:<10} "
            f"label={values['label']} "
            f"valid={values['valid']:<5} "
            f"skipped={values['skipped']:<5} "
            f"mean_z={mean_z:>9.4f} "
            f"flagged={percent_flagged:>7.2f}%"
        )

    print("\nSaved:")
    print("Details:", OUT_JSONL)
    print("Summary:", OUT_CSV)
    print("=" * 70)


if __name__ == "__main__":
    main()