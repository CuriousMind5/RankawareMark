# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# ================= GPU =================
# OPT-13B will be distributed across GPUs 0 and 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

USE_MULTI_GPU = True
DEVICE = "cuda:0"
DTYPE = "float16"
BATCH_SIZE = 4

MAX_MEMORY = {
    0: "22GiB",
    1: "22GiB",
    "cpu": "60GiB",
}


# ================= PATHS =================
INPUT_DIR = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "reviewer/Rankware_entropy"
)

OUT_DIR = (
    "/home/sy/markllm/MarkLLM-main/MarkLLM-main/"
    "reviewer/Rankware_entropy/PPL/Qwen3-8B"
)

MODEL_PATH = (
    "/home/sy/watermark_unlearning_experiment/"
    "models/Qwen3-8B"
)


# ================= EXPERIMENT FILES =================
EXPERIMENTS = [
    {
        "model": "OPT-2.7B",
        "variant": "Vanilla",
        "entropy_tau": None,
        "filename": (
            "vanilla_opt-2.7b_n500_tokens200.json"
        ),
    },
    {
        "model": "OPT-2.7B",
        "variant": "Rank-Aware with entropy",
        "entropy_tau": 0.9,
        "filename": (
            "rankaware_opt-2.7b_"
            "with_entropy_tau0p9_n500_tokens200.json"
        ),
    },
    {
        "model": "OPT-2.7B",
        "variant": "Rank-Aware without entropy",
        "entropy_tau": 0.0,
        "filename": (
            "rankaware_opt-2.7b_"
            "without_entropy_tau0_n500_tokens200.json"
        ),
    },
    {
        "model": "OPT-6.7B",
        "variant": "Vanilla",
        "entropy_tau": None,
        "filename": (
            "vanilla_opt-6.7b_n500_tokens200.json"
        ),
    },
    {
        "model": "OPT-6.7B",
        "variant": "Rank-Aware with entropy",
        "entropy_tau": 0.9,
        "filename": (
            "rankaware_opt-6.7b_"
            "with_entropy_tau0p9_n500_tokens200.json"
        ),
    },
    {
        "model": "OPT-6.7B",
        "variant": "Rank-Aware without entropy",
        "entropy_tau": 0.0,
        "filename": (
            "rankaware_opt-6.7b_"
            "without_entropy_tau0_n500_tokens200.json"
        ),
    },
]


PROMPT_FIELD = "original"
TEXT_FIELD = "sampled"


# =========================================================
# IMPORTS
# =========================================================
import csv
import json
import math
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# TOKENIZER
# =========================================================
def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        use_fast=True,
        local_files_only=True,
    )

    if (
        tokenizer.pad_token_id is None
        and tokenizer.eos_token_id is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# =========================================================
# MODEL
# =========================================================
def load_model(path: str):
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    selected_dtype = dtype_map[DTYPE]

    if USE_MULTI_GPU:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            dtype=selected_dtype,
            device_map="auto",
            max_memory=MAX_MEMORY,
            low_cpu_mem_usage=True,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            dtype=selected_dtype,
            low_cpu_mem_usage=True,
        ).to(DEVICE)

    model.eval()

    return model


# =========================================================
# HELPERS
# =========================================================
def strip_bos(ids: list, bos_token_id):
    if (
        bos_token_id is not None
        and len(ids) > 0
        and ids[0] == bos_token_id
    ):
        return ids[1:]

    return ids


def safe_ppl(cross_entropy: float):
    if not math.isfinite(cross_entropy):
        return None

    try:
        perplexity = math.exp(cross_entropy)

    except OverflowError:
        return None

    if not math.isfinite(perplexity):
        return None

    return perplexity


# =========================================================
# FULL-SEQUENCE CE AND PPL
# =========================================================
@torch.no_grad()
def compute_ce_ppl(
    model,
    tokenizer,
    prompts: list[str],
    continuations: list[str],
    max_context: int,
):
    encoded = tokenizer(
        [
            prompt + continuation
            for prompt, continuation in zip(
                prompts,
                continuations,
            )
        ],
        add_special_tokens=False,
    )

    bos_token_id = tokenizer.bos_token_id
    sequences = []

    for token_ids in encoded["input_ids"]:
        token_ids = strip_bos(
            token_ids,
            bos_token_id,
        )

        if len(token_ids) > max_context:
            token_ids = token_ids[-max_context:]

        sequences.append(
            torch.tensor(
                token_ids,
                dtype=torch.long,
            )
        )

    maximum_length = (
        max(len(sequence) for sequence in sequences)
        - 1
    )

    if maximum_length <= 0:
        return [], []

    pad_token_id = tokenizer.pad_token_id

    input_rows = []
    target_rows = []
    attention_rows = []

    for sequence in sequences:
        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        attention_mask = torch.ones_like(
            input_ids
        )

        padding_length = (
            maximum_length - input_ids.size(0)
        )

        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full(
                    (padding_length,),
                    pad_token_id,
                    dtype=torch.long,
                ),
            ])

            target_ids = torch.cat([
                target_ids,
                torch.full(
                    (padding_length,),
                    pad_token_id,
                    dtype=torch.long,
                ),
            ])

            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(
                    padding_length,
                    dtype=torch.long,
                ),
            ])

        input_rows.append(input_ids)
        target_rows.append(target_ids)
        attention_rows.append(attention_mask)

    # For a sharded model, send the input to the embedding device.
    input_device = (
        model.get_input_embeddings().weight.device
    )

    input_tensor = torch.stack(
        input_rows
    ).to(input_device)

    target_tensor = torch.stack(
        target_rows
    ).to(input_device)

    attention_tensor = torch.stack(
        attention_rows
    ).to(input_device)

    logits = model(
        input_ids=input_tensor,
        attention_mask=attention_tensor,
    ).logits

    log_probabilities = torch.log_softmax(
        logits,
        dim=-1,
    )

    token_log_probabilities = (
        log_probabilities
        .gather(
            -1,
            target_tensor.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    valid_mask = attention_tensor.float()

    negative_log_likelihood = -(
        token_log_probabilities * valid_mask
    ).sum(dim=1)

    token_counts = valid_mask.sum(dim=1)

    cross_entropy = (
        negative_log_likelihood
        / torch.clamp_min(token_counts, 1.0)
    )

    return (
        cross_entropy.cpu().tolist(),
        token_counts.cpu().tolist(),
    )


# =========================================================
# PROCESS ONE FILE
# =========================================================
def process_file(
    model,
    tokenizer,
    experiment: dict,
    batch_size: int,
    max_context: int,
):
    filename = experiment["filename"]
    file_path = os.path.join(
        INPUT_DIR,
        filename,
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing input file: {file_path}"
        )

    file_stem = os.path.splitext(filename)[0]

    output_jsonl = os.path.join(
        OUT_DIR,
        f"{file_stem}__ppl_all.jsonl",
    )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    rows = []

    with open(
        output_jsonl,
        "w",
        encoding="utf-8",
    ) as writer:
        prompt_buffer = []
        continuation_buffer = []
        id_buffer = []

        def process_buffer():
            if not prompt_buffer:
                return

            cross_entropies, token_counts = (
                compute_ce_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompt_buffer,
                    continuations=continuation_buffer,
                    max_context=max_context,
                )
            )

            for position, (
                cross_entropy,
                token_count,
            ) in enumerate(
                zip(
                    cross_entropies,
                    token_counts,
                )
            ):
                perplexity = safe_ppl(
                    cross_entropy
                )

                record = {
                    "model": experiment["model"],
                    "variant": experiment["variant"],
                    "entropy_tau": (
                        experiment["entropy_tau"]
                    ),
                    "source_file": filename,
                    "idx": id_buffer[position],
                    "ce_nats": float(
                        cross_entropy
                    ),
                    "ppl": (
                        float(perplexity)
                        if perplexity is not None
                        else None
                    ),
                    "tokens": int(token_count),
                }

                writer.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                rows.append(record)

        for index, item in enumerate(
            tqdm(
                data,
                desc=file_stem,
            )
        ):
            prompt = str(
                item.get(PROMPT_FIELD, "")
            )

            continuation = str(
                item.get(TEXT_FIELD, "")
            )

            if not continuation.strip():
                continue

            prompt_buffer.append(prompt)
            continuation_buffer.append(
                continuation
            )
            id_buffer.append(
                item.get("id", index)
            )

            if len(prompt_buffer) == batch_size:
                process_buffer()

                prompt_buffer.clear()
                continuation_buffer.clear()
                id_buffer.clear()

        if prompt_buffer:
            process_buffer()

    print("Saved:", output_jsonl)

    return rows


# =========================================================
# SUMMARY
# =========================================================
def save_summary(all_rows: list[dict]):
    grouped = defaultdict(
        lambda: {
            "ce": [],
            "ppl": [],
            "tokens": [],
        }
    )

    metadata = {}

    for row in all_rows:
        if row["ppl"] is None:
            continue

        key = (
            row["model"],
            row["variant"],
        )

        grouped[key]["ce"].append(
            row["ce_nats"]
        )

        grouped[key]["ppl"].append(
            row["ppl"]
        )

        grouped[key]["tokens"].append(
            row["tokens"]
        )

        metadata[key] = row["entropy_tau"]

    output_csv = os.path.join(
        OUT_DIR,
        "rankaware_entropy_ppl_summary.csv",
    )

    with open(
        output_csv,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "variant",
            "entropy_tau",
            "num_samples",
            "mean_tokens",
            "mean_ce_nats",
            "mean_ppl",
            "std_ppl",
            "median_ppl",
        ])

        print(
            "\n"
            "================ PPL SUMMARY ================"
        )

        print(
            f"{'Model':<12} "
            f"{'Variant':<32} "
            f"{'N':>5} "
            f"{'Mean PPL':>12} "
            f"{'Std PPL':>12} "
            f"{'Median':>12}"
        )

        print("-" * 90)

        for experiment in EXPERIMENTS:
            key = (
                experiment["model"],
                experiment["variant"],
            )

            values = grouped[key]

            sample_count = len(values["ppl"])

            if sample_count == 0:
                mean_tokens = float("nan")
                mean_ce = float("nan")
                mean_ppl = float("nan")
                std_ppl = float("nan")
                median_ppl = float("nan")

            else:
                mean_tokens = float(
                    np.mean(values["tokens"])
                )

                mean_ce = float(
                    np.mean(values["ce"])
                )

                mean_ppl = float(
                    np.mean(values["ppl"])
                )

                std_ppl = float(
                    np.std(values["ppl"])
                )

                median_ppl = float(
                    np.median(values["ppl"])
                )

            writer.writerow([
                experiment["model"],
                experiment["variant"],
                experiment["entropy_tau"],
                sample_count,
                f"{mean_tokens:.2f}",
                f"{mean_ce:.6f}",
                f"{mean_ppl:.6f}",
                f"{std_ppl:.6f}",
                f"{median_ppl:.6f}",
            ])

            print(
                f"{experiment['model']:<12} "
                f"{experiment['variant']:<32} "
                f"{sample_count:>5} "
                f"{mean_ppl:>12.4f} "
                f"{std_ppl:>12.4f} "
                f"{median_ppl:>12.4f}"
            )

    print("\nPPL summary saved:", output_csv)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 75)
    print("RANK-AWARE ENTROPY ABLATION PPL")
    print("=" * 75)
    print("Input directory:", INPUT_DIR)
    print("Output directory:", OUT_DIR)
    print("PPL model:", MODEL_PATH)
    print("Multi-GPU:", USE_MULTI_GPU)

    tokenizer = load_tokenizer(MODEL_PATH)
    model = load_model(MODEL_PATH)

    max_context = (
        model.config.max_position_embeddings
    )

    print("Maximum context length:", max_context)

    all_rows = []

    for experiment in EXPERIMENTS:
        all_rows.extend(
            process_file(
                model=model,
                tokenizer=tokenizer,
                experiment=experiment,
                batch_size=BATCH_SIZE,
                max_context=max_context,
            )
        )

    save_summary(all_rows)

    print("\n[DONE] Rank-Aware entropy PPL completed.")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()
