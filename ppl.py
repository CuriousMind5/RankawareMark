# -*- coding: utf-8 -*-


# ================= GPU =================
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
INPUT_DIR = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/reviewer/outputs_c4_realnewslike_Llama-3-8B-Instruct/"
OUT_DIR   = "/home/sy/markllm/MarkLLM-main/MarkLLM-main/reviewer/outputs_c4_realnewslike_Llama-3-8B-Instruct/PPL/opt-13B"

MODEL_PATH = "/home/sy/watermark_unlearning_experiment/models/facebook/opt-13B"

FILES = [
    "vanilla_n500.json",
    "kgw_n500.json",
    "sweet_n500.json",
    "unigram_n500.json",
    "ewd_n500.json",
    "rankaware_n500.json",
]

PROMPT_FIELD = "original"
TEXT_FIELD   = "sampled"

# ==================================================
import os, json, math
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- TOKENIZER ----------------
def load_tok(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

# ---------------- MODEL ----------------
def load_model(path):
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if USE_MULTI_GPU:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype_map[DTYPE],
            device_map="auto",
            max_memory=MAX_MEMORY,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=dtype_map[DTYPE],
            low_cpu_mem_usage=True,
        ).to(DEVICE)

    model.eval()
    return model

# ---------------- HELPERS ----------------
def strip_bos(ids, bos):
    if bos is not None and len(ids) > 0 and ids[0] == bos:
        return ids[1:]
    return ids

# ---------------- FULL-SEQUENCE CE/PPL ----------------
@torch.no_grad()
def compute_ce_ppl(model, tok, prompts, continuations, max_ctx):

    enc = tok(
        [p + c for p, c in zip(prompts, continuations)],
        add_special_tokens=False
    )

    bos = tok.bos_token_id
    seqs = []

    for ids in enc["input_ids"]:
        ids = strip_bos(ids, bos)

        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    T = max(len(s) for s in seqs) - 1
    if T <= 0:
        return [], []

    pad = tok.pad_token_id
    xs, ys, ms = [], [], []

    for s in seqs:
        x = s[:-1]
        y = s[1:]

        m = torch.ones_like(x)

        pad_len = T - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), pad)])
            y = torch.cat([y, torch.full((pad_len,), pad)])
            m = torch.cat([m, torch.zeros(pad_len)])

        xs.append(x)
        ys.append(y)
        ms.append(m)

    device = next(model.parameters()).device
    X = torch.stack(xs).to(device)
    Y = torch.stack(ys).to(device)
    M = torch.stack(ms).to(device)

    logits = model(input_ids=X, attention_mask=M).logits
    logprob = torch.log_softmax(logits, dim=-1)

    tok_lp = logprob.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
    mask = M.float()

    nll = -(tok_lp * mask).sum(dim=1)
    N = mask.sum(dim=1)

    ce = nll / torch.clamp_min(N, 1.0)

    return ce.cpu().tolist(), N.cpu().tolist()

# ---------------- FILE PROCESS ----------------
def process_file(model, tok, file_path, batch, max_ctx):

    name = os.path.basename(file_path).replace(".json", "")
    out_json = os.path.join(OUT_DIR, f"{name}__ppl_all.jsonl")
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(out_json, "w", encoding="utf-8") as w:

        buf_p, buf_c, buf_i = [], [], []

        for obj in tqdm(data, desc=name):

            p = obj.get(PROMPT_FIELD, "")
            c = obj.get(TEXT_FIELD, "")

            if not c.strip():
                continue

            buf_p.append(p)
            buf_c.append(c)
            buf_i.append(obj.get("id", 0))

            if len(buf_p) == batch:
                ces, toks = compute_ce_ppl(model, tok, buf_p, buf_c, max_ctx)

                for j, (ce, tokc) in enumerate(zip(ces, toks)):
                    ppl = math.exp(ce) if math.isfinite(ce) else None

                    w.write(json.dumps({
                        "method": name,
                        "idx": int(buf_i[j]),
                        "ce_nats": float(ce),
                        "ppl": float(ppl) if ppl else None,
                        "tokens": int(tokc),
                    }) + "\n")

                    rows.append((name, tokc, ce, ppl))

                buf_p, buf_c, buf_i = [], [], []

        if buf_p:
            ces, toks = compute_ce_ppl(model, tok, buf_p, buf_c, max_ctx)

            for j, (ce, tokc) in enumerate(zip(ces, toks)):
                ppl = math.exp(ce) if math.isfinite(ce) else None

                w.write(json.dumps({
                    "method": name,
                    "idx": int(buf_i[j]),
                    "ce_nats": float(ce),
                    "ppl": float(ppl) if ppl else None,
                    "tokens": int(tokc),
                }) + "\n")

                rows.append((name, tokc, ce, ppl))

    print("Saved:", out_json)
    return rows

# ---------------- MAIN ----------------
def main():

    tok = load_tok(MODEL_PATH)
    model = load_model(MODEL_PATH)

    all_rows = []

    for fname in FILES:
        path = os.path.join(INPUT_DIR, fname)
        assert os.path.exists(path), f"Missing file: {path}"

        all_rows.extend(
            process_file(model, tok, path, BATCH_SIZE,
                         model.config.max_position_embeddings)
        )

    stats_ce, stats_ppl, stats_tok = defaultdict(list), defaultdict(list), defaultdict(list)

    for method, tokc, ce, ppl in all_rows:
        if ppl is None:
            continue
        stats_ce[method].append(ce)
        stats_ppl[method].append(ppl)
        stats_tok[method].append(tokc)

    out_csv = os.path.join(OUT_DIR, "ppl_full_summary.csv")

    with open(out_csv, "w", encoding="utf-8") as w:
        w.write("method,num_samples,mean_tokens,mean_ce_nats,mean_ppl\n")

        for m in FILES:
            key = m.replace(".json", "")
            w.write(
                f"{key},{len(stats_ppl[key])},"
                f"{np.mean(stats_tok[key]):.2f},"
                f"{np.mean(stats_ce[key]):.6f},"
                f"{np.mean(stats_ppl[key]):.6f}\n"
            )

    print("\n[DONE] Full-sequence PPL summary →", out_csv)

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
