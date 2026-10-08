# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import torch
import warnings
from tqdm import tqdm
import nltk

from transformers import MarianMTModel, MarianTokenizer

# ---------------- QUIET ----------------
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- GPU SETUP ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------- NLTK ----------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------- LOCAL MODEL PATHS ----------------
MODEL_EN_DE = "/home/sy/watermark_unlearning_experiment/models/Helsinki-NLP/opus-mt-en-fr/"
MODEL_DE_EN = "/home/sy/watermark_unlearning_experiment/models/Helsinki-NLP/opus-mt-fr-en/"

# ---------------- DATA PATHS ----------------
INPUT_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_cnn_dailymail_test_1k_opt-6.7B_entropy_0.9"
OUT_DIR   = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_cnn_dailymail_test_1k_opt-6.7B_entropy_0.9/English-French-English"

# ---------------- GENERATION PARAMS (FROM IMAGE) ----------------
GEN_KWARGS = dict(
    num_beams=2,
    do_sample=True,
    temperature=0.7,
    top_p=0.92,
    top_k=0,
    max_new_tokens=280,
    repetition_penalty=1.05,
    no_repeat_ngram_size=4,
    early_stopping=True,
)

MAX_INPUT_LEN = 512

# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- MAIN ----------------
def main():

    ensure_dir(OUT_DIR)

    print(">>> Loading LOCAL OPUS translation models on", DEVICE)

    tok1 = MarianTokenizer.from_pretrained(MODEL_EN_DE, local_files_only=True)
    mod1 = MarianMTModel.from_pretrained(
        MODEL_EN_DE,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    tok2 = MarianTokenizer.from_pretrained(MODEL_DE_EN, local_files_only=True)
    mod2 = MarianMTModel.from_pretrained(
        MODEL_DE_EN,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])
    print(">>> Translating files:", files)

    for fname in files:

        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(
            OUT_DIR, fname.replace(".json", "_translation_attack_sampling.json")
        )

        print(">>> Processing:", fname)

        data = load_json(in_path)
        new_records = []

        for rec in tqdm(data, desc=fname):

            text = rec.get("sampled", "").strip()
            rid = rec.get("id", -1)

            if not text:
                continue

            # -------- EN → DE --------
            batch = tok1(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LEN,
            )
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.inference_mode():
                out_mid = mod1.generate(**batch, **GEN_KWARGS)

            mid_text = tok1.batch_decode(out_mid, skip_special_tokens=True)

            # -------- DE → EN --------
            batch = tok2(
                mid_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LEN,
            )
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.inference_mode():
                out_final = mod2.generate(**batch, **GEN_KWARGS)

            attacked = tok2.batch_decode(out_final, skip_special_tokens=True)[0]

            new_records.append({
                "id": rid,
                "original": text,
                "sampled": attacked.strip(),
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)

    print("DONE")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    torch.manual_seed(15485863)
    main()
