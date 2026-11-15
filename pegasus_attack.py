# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import torch
import warnings
from tqdm import tqdm
import nltk
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# ---------------- QUIET ----------------
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- NLTK ----------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------- PATHS ----------------
PEGASUS_PATH = "/home/sy/watermark_unlearning_experiment/models/pegasus/"

# INPUT = watermarked generation outputs (JSON list of dicts)
INPUT_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_cnn_dailymail_test_1k_opt-6.7B_entropy_0.9"

# OUTPUT = paraphrased attack outputs (JSON list of dicts)
OUT_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/outputs_cnn_dailymail_test_1k_opt-6.7B_entropy_0.9/Pegasus"

# ---------------- PARAMS (MATCH ORIGINAL ATTACK CODE) ----------------
MAX_LEN = 60
NUM_BEAMS = 25
NUM_RETURN = 25
TEMPERATURE = 1.5

KEEP_FINAL = 5   # final paraphrases per continuation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- HELPERS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paraphrase_sentence_multi(model, tokenizer, sent):
    """Return 25 paraphrases (original attack behavior)."""
    batch = tokenizer(
        [sent],
        truncation=True,
        padding="longest",
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **batch,
            max_length=MAX_LEN,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_RETURN,
            temperature=TEMPERATURE,
        )

    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts  # length = 25


# ---------------- MAIN ----------------
def main():

    ensure_dir(OUT_DIR)

    print(">>> Loading Pegasus (local)")
    tokenizer = PegasusTokenizer.from_pretrained(PEGASUS_PATH, local_files_only=True)
    model = PegasusForConditionalGeneration.from_pretrained(
        PEGASUS_PATH,
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

            watermarked_text = rec["sampled"]   # BEFORE attack
            rid = rec.get("id", -1)

            if not watermarked_text.strip():
                continue

            sents = nltk.sent_tokenize(watermarked_text)

            para_versions = [""] * KEEP_FINAL

            for s in sents:
                all_paraphrases = paraphrase_sentence_multi(model, tokenizer, s)  # 25
                outs = all_paraphrases[::5]  # take every 5th → 5 paraphrases (exact old logic)

                for i in range(KEEP_FINAL):
                    para_versions[i] += (" " + outs[i])

            for i in range(KEEP_FINAL):
                new_records.append({
                    "id": rid,
                    "original": watermarked_text.strip(),   # ✅ watermarked
                    "sampled": para_versions[i].strip(),    # ✅ paraphrased
                })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, ensure_ascii=False, indent=2)

        print("Saved:", out_path)

    print("DONE")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()

