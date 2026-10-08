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
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.transformers_config import TransformersConfig
from watermark.rankaware import RankAware, RankAwareConfig


# ================= CLI =================
parser = argparse.ArgumentParser()

parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--prompt_tokens", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--seed", type=int, default=42)

# Existing files are skipped unless this flag is supplied.
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Regenerate and overwrite existing output files.",
)

args = parser.parse_args()


# ================= DEVICE =================
DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Running on device:", DEVICE)


# ================= PATHS =================
BASE = Path(
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main"
)

INPUT_JSON = Path(
    "/home/sy/watermark_unlearning_experiment/data/"
    "c4_realnewslike.json"
)

RANKAWARE_CONFIG_PATH = BASE / "config/RankAware.json"

OUT_DIR = BASE / "reviewer/Rankware_entropy"
OUT_DIR.mkdir(exist_ok=True, parents=True)


# ================= MODELS =================
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


# ================= RANK-AWARE VARIANTS =================
VARIANTS = [
    {
        "name": "with_entropy_tau0p9",
        "entropy_tau": 0.9,
        "entropy_gate_enabled": True,
    },
    {
        "name": "without_entropy_tau0",
        "entropy_tau": 0.0,
        "entropy_gate_enabled": False,
    },
]


# ================= SANITY CHECKS =================
if not INPUT_JSON.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_JSON}"
    )

if not RANKAWARE_CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"RankAware config not found: "
        f"{RANKAWARE_CONFIG_PATH}"
    )

for model_info in MODELS:
    model_path = model_info["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}"
        )

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"config.json missing from: {model_path}"
        )


# ================= HELPERS =================
def reset_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_any_json(path: Path) -> list:
    records = []

    with open(path, "r", encoding="utf-8") as file:
        first_character = file.read(1)
        file.seek(0)

        if first_character == "[":
            print("Detected JSON array format")
            records = json.load(file)

        else:
            print("Detected JSONL format")

            for line in file:
                line = line.strip()

                if line:
                    records.append(json.loads(line))

    return records


def extract_text(item: dict) -> str:
    for key in [
        "text",
        "document",
        "article",
        "content",
        "prefix",
    ]:
        if key in item:
            return item[key]

    if "question" in item:
        question = item["question"]

        if item.get("human_answers"):
            return (
                question + " " + item["human_answers"][0]
            )

        if item.get("chatgpt_answers"):
            return (
                question + " " + item["chatgpt_answers"][0]
            )

        return question

    raise ValueError(
        f"Unknown dataset format. Keys: {list(item.keys())}"
    )


def get_prompt_tokens(
    tokenizer,
    text: str,
    number_of_tokens: int,
) -> str:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"][:number_of_tokens]

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )


def strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].lstrip()

    return full_text


def save_json(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )


def should_generate(path: Path) -> bool:
    if path.exists() and not args.overwrite:
        print("Already exists, skipping:", path)
        return False

    return True


def vanilla_output_path(model_name: str) -> Path:
    return OUT_DIR / (
        f"vanilla_{model_name}_"
        f"n{len(data)}_"
        f"tokens{args.max_new_tokens}.json"
    )


def rankaware_output_path(
    model_name: str,
    variant_name: str,
) -> Path:
    return OUT_DIR / (
        f"rankaware_{model_name}_{variant_name}_"
        f"n{len(data)}_"
        f"tokens{args.max_new_tokens}.json"
    )


def build_rankaware(
    tf_cfg,
    entropy_tau: float,
):
    """
    Load the original RankAware configuration and override
    only entropy_tau. RankAware.json is not modified.
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

    # Keep the stored configuration dictionary consistent.
    if isinstance(config.config_dict, dict):
        config.config_dict["entropy_tau"] = float(
            entropy_tau
        )

    print(
        "Entropy tau override:",
        f"{original_tau} -> {config.entropy_tau}",
    )

    watermark = RankAware(
        config,
        tf_cfg,
    )

    return config, watermark


# ================= LOAD DATASET =================
data = load_any_json(INPUT_JSON)

print("Total dataset samples:", len(data))

if args.num_samples > len(data):
    raise ValueError(
        f"Requested {args.num_samples} samples, "
        f"but the dataset contains only {len(data)}."
    )

data = data[:args.num_samples]

print("Selected samples:", len(data))


# ================= RUN EACH MODEL =================
for model_info in MODELS:
    model_name = model_info["name"]
    model_path = model_info["path"]

    vanilla_path = vanilla_output_path(model_name)

    rankaware_paths = [
        rankaware_output_path(
            model_name,
            variant["name"],
        )
        for variant in VARIANTS
    ]

    required_paths = [
        vanilla_path,
        *rankaware_paths,
    ]

    # Avoid loading a large model when all its outputs exist.
    if (
        not args.overwrite
        and all(path.exists() for path in required_paths)
    ):
        print("\nAll outputs already exist for:", model_name)

        for path in required_paths:
            print("Existing:", path)

        continue

    print("\n" + "#" * 75)
    print("Loading model:", model_name)
    print("Model path:", model_path)
    print("#" * 75)

    # ================= TOKENIZER =================
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================= MODEL =================
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.float16,
    ).to(DEVICE)

    model.eval()

    model.generation_config.pad_token_id = (
        tokenizer.eos_token_id
    )

    # ================= TRANSFORMERS CONFIG =================
    tf_cfg = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        vocab_size=(
            model.get_output_embeddings().weight.shape[0]
        ),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
    )

    tf_cfg.gen_kwargs["pad_token_id"] = (
        tokenizer.eos_token_id
    )

    # =====================================================
    # 1. GENERATE VANILLA
    # =====================================================
    if should_generate(vanilla_path):
        print("\n" + "=" * 75)
        print("Model:", model_name)
        print("Generating: vanilla")
        print("=" * 75)

        reset_seed(args.seed)

        vanilla_records = []

        for index, item in enumerate(
            tqdm(
                data,
                desc=f"{model_name}-vanilla",
            )
        ):
            source_text = extract_text(item)

            prompt = get_prompt_tokens(
                tokenizer=tokenizer,
                text=source_text,
                number_of_tokens=args.prompt_tokens,
            )

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(DEVICE)

            prompt_length = encoded["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    **tf_cfg.gen_kwargs,
                )

            continuation_ids = output_ids[
                0,
                prompt_length:,
            ]

            continuation = tokenizer.decode(
                continuation_ids,
                skip_special_tokens=True,
            ).strip()

            vanilla_records.append(
                {
                    "id": index,
                    "original": prompt,
                    "sampled": continuation,
                    "method": "Vanilla",
                    "model": model_name,
                }
            )

        save_json(
            vanilla_path,
            vanilla_records,
        )

        print("Saved:", vanilla_path)

    # =====================================================
    # 2. GENERATE RANK-AWARE VARIANTS
    # =====================================================
    for variant in VARIANTS:
        variant_name = variant["name"]
        entropy_tau = variant["entropy_tau"]
        entropy_gate_enabled = variant[
            "entropy_gate_enabled"
        ]

        output_path = rankaware_output_path(
            model_name,
            variant_name,
        )

        if not should_generate(output_path):
            continue

        print("\n" + "=" * 75)
        print("Model:", model_name)
        print("Generating:", variant_name)
        print("Entropy tau:", entropy_tau)
        print("=" * 75)

        # Reset to the same initial random state.
        reset_seed(args.seed)

        rankaware_config, rankaware = build_rankaware(
            tf_cfg=tf_cfg,
            entropy_tau=entropy_tau,
        )

        rankaware_records = []

        for index, item in enumerate(
            tqdm(
                data,
                desc=f"{model_name}-{variant_name}",
            )
        ):
            source_text = extract_text(item)

            prompt = get_prompt_tokens(
                tokenizer=tokenizer,
                text=source_text,
                number_of_tokens=args.prompt_tokens,
            )

            full_generation = (
                rankaware.generate_watermarked_text(
                    prompt
                )
            )

            continuation = strip_prompt(
                full_text=full_generation,
                prompt=prompt,
            )

            rankaware_records.append(
                {
                    "id": index,
                    "original": prompt,
                    "sampled": continuation,
                    "method": "RankAware",
                    "model": model_name,
                    "entropy_tau": entropy_tau,
                    "entropy_gate_enabled": (
                        entropy_gate_enabled
                    ),
                }
            )

        save_json(
            output_path,
            rankaware_records,
        )

        print("Saved:", output_path)

        del rankaware
        del rankaware_config

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================= UNLOAD CURRENT MODEL =================
    del tf_cfg
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\nRank-Aware entropy ablation completed.")
print("Output directory:", OUT_DIR)