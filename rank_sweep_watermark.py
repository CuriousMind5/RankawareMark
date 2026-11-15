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
from watermark.rankaware import RankAwareKWG, RankAwareKWGConfig


# ================= OFFLINE =================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================= CLI =================
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--prompt_tokens", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()


# ================= PATHS =================
BASE = Path("/home/sy/markllm/MarkLLM-main/MarkLLM-main")
MODEL_PATH = Path("/home/sy/watermark_unlearning_experiment/models/facebook/opt-6.7B/")
INPUT_JSON = Path("/home/sy/watermark_unlearning_experiment/data/c4_realnewslike.json")

CONFIG_PATH = BASE / "config/RankAwareKWG.json"
ROOT_OUT = BASE / f"outputs_{INPUT_JSON.stem}_opt-6.7_entropy_0.9/rank_decay"

ROOT_OUT.mkdir(parents=True, exist_ok=True)


# ================= DATA LOADER =================
def load_any_json(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
        else:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def extract_text(item):
    for k in ["text", "document", "article", "content", "prefix"]:
        if k in item:
            return item[k]
    raise ValueError("Unknown dataset format")


def get_prompt(text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:args.prompt_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def strip_prompt(full, prompt):
    if full.startswith(prompt):
        return full[len(prompt):].lstrip()
    return full


def save_json(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# ================= LOAD DATA =================
data = load_any_json(INPUT_JSON)[: args.num_samples]


# ================= MODEL =================
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()
DEVICE = model.device


# ================= TRANSFORMERS CONFIG =================
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


# ====================================================
# RANK-AWARE SWEEP — FILE-DRIVEN CLEAN RELOAD
# ====================================================
rank_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

for r in rank_values:

    print("\n==============================")
    print(f" Running RankAware r0 = {r}")
    print("==============================")

    # fresh config load from file
    cfg = RankAwareKWGConfig(CONFIG_PATH, tf_cfg)

    # override only rank
    cfg.r0 = float(r)

    # rebuild watermark instance
    wm = RankAwareKWG(cfg, tf_cfg)

    # experiment folder
    EXP_DIR = ROOT_OUT / f"vanilla_vs_rank{r}"
    EXP_DIR.mkdir(exist_ok=True)

    vanilla_records = []
    wm_records = []

    for i, item in enumerate(tqdm(data, desc=f"rank={r}")):

        text = extract_text(item)
        prompt = get_prompt(text)

        # ---------- vanilla ----------
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = model.generate(**enc, **tf_cfg.gen_kwargs)

        full_v = tokenizer.decode(out[0], skip_special_tokens=True)
        cont_v = strip_prompt(full_v, prompt)

        vanilla_records.append({
            "id": i,
            "original": prompt,
            "sampled": cont_v,
        })

        # ---------- watermark ----------
        full_w = wm.generate_watermarked_text(prompt)
        cont_w = strip_prompt(full_w, prompt)

        wm_records.append({
            "id": i,
            "original": prompt,
            "sampled": cont_w,
        })

    save_json(EXP_DIR / "vanilla.json", vanilla_records)
    save_json(EXP_DIR / "rankaware.json", wm_records)

    print("Saved experiment →", EXP_DIR)


print("\nALL RANK PAIRS GENERATED SUCCESSFULLY.")
