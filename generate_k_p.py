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

# ================= OFFLINE =================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= CLI =================
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=250)
parser.add_argument("--prompt_tokens", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()

# ================= PATHS =================
BASE = Path("/home/sy/markllm/MarkLLM-main/MarkLLM-main")
MODEL_PATH = Path("/home/sy/watermark_unlearning_experiment/models/facebook/opt-2.7b")
INPUT_JSON = Path("/home/sy/watermark_unlearning_experiment/data/c4_realnewslike.json")

OUT_DIR = BASE / "outputs_fastdetectgpt_opt-2.7B_250_json"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ================= SANITY CHECK =================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
if not (MODEL_PATH / "config.json").exists():
    raise FileNotFoundError(f"config.json missing in: {MODEL_PATH}")

# ================= LOAD DATA =================
data = []
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    first = f.read(1)
    f.seek(0)
    if first == "[":  # Check if JSON is a list
        data = json.load(f)
    else:  # If it's in JSONL format
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

data = data[: args.num_samples]
print("Loaded samples:", len(data))
print("OUT_DIR:", OUT_DIR)

# ================= LOAD MODEL =================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
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

# Set top-k and top-p in gen_kwargs
tf_cfg.gen_kwargs = {
    'top_k': 100,  # Apply top-k sampling with k = 100
    'top_p': 0.95,  # Apply top-p (nucleus) sampling with p = 0.95
    'do_sample': True,  # Enable sampling
    'temperature': args.temperature,
    'max_new_tokens': args.max_new_tokens
}

# ================= WATERMARK OBJECTS =================
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

# ================= HELPERS =================
def get_prompt_tokens(text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].lstrip()
    return full_text

def save_json(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# ================= VANILLA =================
print("\n===== Running vanilla =====")
vanilla_out = OUT_DIR / f"vanilla_n{len(data)}.json"
vanilla_records = []

for i, item in enumerate(tqdm(data, desc="vanilla")):
    text = item["text"]
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

save_json(vanilla_out, vanilla_records)
print("Saved:", vanilla_out)

# ================= WATERMARKED =================
for name, wm in methods.items():
    print(f"\n===== Running {name} =====")
    out_path = OUT_DIR / f"{name}_n{len(data)}.json"
    records = []

    for i, item in enumerate(tqdm(data, desc=name)):
        text = item["text"]
        prompt = get_prompt_tokens(text, args.prompt_tokens)

        full_gen = wm.generate_watermarked_text(prompt)
        continuation = strip_prompt(full_gen, prompt)

        records.append({
            "id": i,
            "original": prompt,
            "sampled": continuation,
        })

    save_json(out_path, records)
    print("Saved:", out_path)

print("\nALL DONE.")

