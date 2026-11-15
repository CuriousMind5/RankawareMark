# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.transformers_config import TransformersConfig

from watermark.kgw.kgw import KGW, KGWConfig
from watermark.unigram.unigram import Unigram, UnigramConfig
from watermark.sweet.sweet import SWEET, SWEETConfig
from watermark.ewd.ewd import EWD, EWDConfig
from watermark.rankaware import RankAwareKWG, RankAwareKWGConfig


# =========================================================
# OFFLINE SETTINGS
# =========================================================

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================================================
# CLI ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--prompt_tokens", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()


# =========================================================
# PATHS
# =========================================================

BASE = Path("/home/sy/markllm/MarkLLM-main/MarkLLM-main")
MODEL_PATH = Path("/home/sy/watermark_unlearning_experiment/models/facebook/opt-6.7B/")
INPUT_JSON = Path("/home/sy/watermark_unlearning_experiment/data/c4_realnewslike.json")

OUT_DIR = BASE / f"outputs_{INPUT_JSON.stem}_opt-6.7B_entropy_0.9_Gamma_sweep"
OUT_DIR.mkdir(exist_ok=True, parents=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_any_json(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def extract_text(item: dict) -> str:
    if "text" in item:
        return item["text"]
    if "document" in item:
        return item["document"]
    if "article" in item:
        return item["article"]
    if "content" in item:
        return item["content"]
    if "question" in item:
        q = item["question"]
        if "human_answers" in item and item["human_answers"]:
            return q + " " + item["human_answers"][0]
        if "chatgpt_answers" in item and item["chatgpt_answers"]:
            return q + " " + item["chatgpt_answers"][0]
        return q
    if "prefix" in item:
        return item["prefix"]
    raise ValueError(f"Unknown format: {list(item.keys())}")


data = load_any_json(INPUT_JSON)[:args.num_samples]
print("Total loaded:", len(data))


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================================================
# MODEL
# =========================================================

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

DEVICE = model.device
model.generation_config.pad_token_id = tokenizer.eos_token_id


# =========================================================
# TRANSFORMERS CONFIG
# =========================================================

tf_cfg = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    vocab_size=model.get_output_embeddings().weight.shape[0],
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    do_sample=True,
)
tf_cfg.gen_kwargs["pad_token_id"] = tokenizer.eos_token_id


# =========================================================
# WATERMARK OBJECTS (FROM CONFIG JSON)
# =========================================================

kgw = KGW(KGWConfig(BASE / "config/KGW.json", tf_cfg), tf_cfg)
uni = Unigram(UnigramConfig(BASE / "config/Unigram.json", tf_cfg), tf_cfg)
sweet = SWEET(SWEETConfig(BASE / "config/SWEET.json", tf_cfg), tf_cfg)
ewd = EWD(EWDConfig(BASE / "config/EWD.json", tf_cfg), tf_cfg)
rankaware = RankAwareKWG(RankAwareKWGConfig(BASE / "config/RankAwareKWG.json", tf_cfg), tf_cfg)

methods = {
    "kgw": kgw,
    "unigram": uni,
    "sweet": sweet,
    "ewd": ewd,
    "rankaware": rankaware,
}


# =========================================================
# HELPERS
# =========================================================

def get_prompt_tokens(text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].lstrip()
    return full_text


def save_json(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# =========================================================
# GAMMA SWEEP
# =========================================================

GAMMA_LIST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]

for gamma_value in GAMMA_LIST:

    print("\n========================================")
    print(f"Running GAMMA = {gamma_value}")
    print("========================================")

    gamma_folder = OUT_DIR / f"gamma_{gamma_value}"
    gamma_folder.mkdir(exist_ok=True)

    # ---------------- VANILLA ----------------
    print("Running vanilla...")
    vanilla_records = []

    for i, item in enumerate(tqdm(data, desc=f"vanilla_g{gamma_value}")):
        text = extract_text(item)
        prompt = get_prompt_tokens(text, args.prompt_tokens)

        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out_ids = model.generate(**enc, **tf_cfg.gen_kwargs)

        full_gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        continuation = strip_prompt(full_gen, prompt)

        vanilla_records.append({
            "id": i,
            "original": prompt,
            "sampled": continuation,
        })

    save_json(gamma_folder / f"vanilla_n{len(data)}.json", vanilla_records)

    # ---------------- WATERMARK METHODS ----------------
    for name, wm in methods.items():

        print(f"\n----- Running {name} | gamma={gamma_value} -----")

        # Override ONLY gamma
        if hasattr(wm, "gamma"):
            wm.gamma = gamma_value
        if hasattr(wm, "config") and hasattr(wm.config, "gamma"):
            wm.config.gamma = gamma_value

        records = []

        for i, item in enumerate(tqdm(data, desc=f"{name}_g{gamma_value}")):
            text = extract_text(item)
            prompt = get_prompt_tokens(text, args.prompt_tokens)

            full_gen = wm.generate_watermarked_text(prompt)
            continuation = strip_prompt(full_gen, prompt)

            records.append({
                "id": i,
                "gamma": gamma_value,
                "original": prompt,
                "sampled": continuation,
            })

        save_json(gamma_folder / f"{name}.json", records)


print("\nALL DONE.")
