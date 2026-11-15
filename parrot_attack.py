# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import torch
import warnings
from tqdm import tqdm
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- QUIET ----------------
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- NLTK ----------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------- PATHS ----------------
PARROT_MODEL_PATH = "/home/sy/watermark_unlearning_experiment/models/parrot_pharaphraser_on_T5/"

# INPUT = watermarked generation outputs
INPUT_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_cnn_dailymail_test_1k_opt-6.7B_entropy_0.9"

# OUTPUT = paraphrased attack outputs
OUT_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_cnn_dailymail_test_1k_opt-6.7B_entropy_0.9/Parrot"

# ---------------- PARAMS (EXACT SAME STYLE) ----------------
MAX_LEN = 32
MAX_RETURN = 10
TOP_K = 50
TOP_P = 0.95
DO_DIVERSE = False   # kept for consistency, not used

KEEP_FINAL = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paraphrase_sentence_multi(model, tokenizer, sent):
    sent = sent.strip()
    if not sent:
        return []

    input_text = "paraphrase: " + sent

    batch = tokenizer(
        [input_text],
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **batch,
            do_sample=True,
            top_k=TOP_K,
            top_p=TOP_P,
            max_length=MAX_LEN,
            num_return_sequences=MAX_RETURN,
            early_stopping=True,
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts


# ---------------- MAIN ----------------
def main():

    ensure_dir(OUT_DIR)

    print(">>> Loading Parrot-T5 model (local)")

    tokenizer = AutoTokenizer.from_pretrained(PARROT_MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        PARROT_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    model.eval()

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json")])
    print(">>> Paraphrasing files:", files)

    for fname in files:

        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname.replace(".json", "_para.json"))

        print(">>> Processing:", fname)

        data = load_json(in_path)
        new_records = []

        for rec in tqdm(data, desc=fname):

            watermarked_text = rec["sampled"]
            rid = rec.get("id", -1)

            if not watermarked_text.strip():
                continue

            sents = nltk.sent_tokenize(watermarked_text)

            para_versions = [""] * KEEP_FINAL

            for s in sents:
                try:
                    paras = paraphrase_sentence_multi(model, tokenizer, s)
                except Exception:
                    paras = []

                if not paras:
                    paras = [s] * KEEP_FINAL

                paras = paras[:KEEP_FINAL]

                for i in range(len(paras)):
                    para_versions[i] += (" " + paras[i])

            for i in range(len(para_versions)):
                new_records.append({
                    "id": rid,
                    "original": watermarked_text.strip(),   # watermarked
                    "sampled": para_versions[i].strip(),    # paraphrased
                })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)

    print("DONE")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()
