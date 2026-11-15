# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# ================= OFFLINE =================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.transformers_config import TransformersConfig

from watermark.kgw.kgw import KGW, KGWConfig
from watermark.unigram.unigram import Unigram, UnigramConfig
from watermark.sweet.sweet import SWEET, SWEETConfig
from watermark.ewd.ewd import EWD, EWDConfig
from watermark.rankaware import RankAware, RankAwareConfig


# ================= CLI =================
parser = argparse.ArgumentParser()

parser.add_argument("--num_samples", type=int, default=500)
parser.add_argument("--prompt_tokens", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)

# -------------------------------------------------
# MODEL PATH
#   OPT-2.7B
#   OPT-6.7B
#   Llama-3-8B-Instruct
#   Qwen3-8B
#   Qwen2.5-Instruct
#   etc.
#
# The script automatically decides whether it
# should use plain continuation or instruction mode.
# -------------------------------------------------
parser.add_argument(
    "--model_path",
    type=str,
    default=(
        "/home/sy/watermark_unlearning_experiment/"
        "models/facebook/Llama-3.1-8B/"
    ),
)

parser.add_argument(
    "--input_json",
    type=str,
    default=(
        "/home/sy/watermark_unlearning_experiment/"
        "data/c4_realnewslike.json"
    ),
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

MODEL_PATH = Path(args.model_path)

INPUT_JSON = Path(args.input_json)


# ================= MODEL TAG =================
def clean_model_name(path: Path) -> str:
    name = path.name

    if not name:
        name = path.parent.name

    return name.replace("/", "_").replace(" ", "_")


MODEL_TAG = clean_model_name(MODEL_PATH)


OUT_DIR = BASE / (
    f"reviewer/"
    f"outputs_{INPUT_JSON.stem}_{MODEL_TAG}"
)

OUT_DIR.mkdir(
    exist_ok=True,
    parents=True,
)


# ================= SANITY =================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MODEL_PATH not found: {MODEL_PATH}"
    )

if not (MODEL_PATH / "config.json").exists():
    raise FileNotFoundError(
        f"config.json missing in: {MODEL_PATH}"
    )


# ================= LOAD JSON / JSONL =================
def load_any_json(path: Path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        first = f.read(1)
        f.seek(0)

        if first == "[":

            print("Detected JSON array format")

            data = json.load(f)

        else:

            print("Detected JSONL format")

            for line in f:

                line = line.strip()

                if not line:
                    continue

                data.append(
                    json.loads(line)
                )

    return data


# ================= EXTRACT TEXT =================
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

        if (
            "human_answers" in item
            and len(item["human_answers"]) > 0
        ):
            return (
                q
                + " "
                + item["human_answers"][0]
            )

        if (
            "chatgpt_answers" in item
            and len(item["chatgpt_answers"]) > 0
        ):
            return (
                q
                + " "
                + item["chatgpt_answers"][0]
            )

        return q

    if "prefix" in item:
        return item["prefix"]

    raise ValueError(
        f"Unknown format. Keys: {list(item.keys())}"
    )


# ================= LOAD DATA =================
data = load_any_json(INPUT_JSON)

print("Total loaded:", len(data))

data = data[:args.num_samples]

print("Samples selected:", len(data))


# ================= TOKENIZER =================
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)


if tokenizer.pad_token_id is None:

    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    else:
        raise RuntimeError(
            "Tokenizer has neither pad_token nor eos_token."
        )




chat_template = getattr(
    tokenizer,
    "chat_template",
    None,
)

IS_INSTRUCTION_MODEL = bool(
    isinstance(chat_template, str)
    and chat_template.strip()
)


print("\n========================================")

if IS_INSTRUCTION_MODEL:

    print("MODEL MODE: INSTRUCTION / CHAT")

    print(
        "Tokenizer contains a chat template."
    )

    print(
        "Generation will use:"
    )

    print(
        "instruction + source passage + chat template"
    )

else:

    print("MODEL MODE: BASE / CONTINUATION")

    print(
        "Tokenizer has no chat template."
    )

    print(
        "Generation will use:"
    )

    print(
        "raw 50-token passage directly"
    )

print("========================================")


# ================= MODEL =================
print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    torch_dtype=torch.float16,
).to(DEVICE)

model.eval()

model.generation_config.pad_token_id = (
    tokenizer.pad_token_id
)


terminators = []

if tokenizer.eos_token_id is not None:

    if isinstance(
        tokenizer.eos_token_id,
        list,
    ):

        terminators.extend(
            tokenizer.eos_token_id
        )

    else:

        terminators.append(
            tokenizer.eos_token_id
        )




if IS_INSTRUCTION_MODEL:

    possible_end_tokens = [
        "<|eot_id|>",
        "<|im_end|>",
    ]

    vocab = tokenizer.get_vocab()

    for special_token in possible_end_tokens:

        if special_token in vocab:

            token_id = tokenizer.convert_tokens_to_ids(
                special_token
            )

            if (
                token_id is not None
                and token_id >= 0
                and token_id not in terminators
            ):

                terminators.append(
                    token_id
                )


print(
    "Generation EOS terminators:",
    terminators,
)


# ================= TRANSFORMERS CONFIG =================
tf_cfg = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    vocab_size=(
        model
        .get_output_embeddings()
        .weight
        .shape[0]
    ),

    # -----------------------------------------
    # EXACT requested generated length
    # -----------------------------------------
    max_new_tokens=args.max_new_tokens,
    min_new_tokens=args.max_new_tokens,

    temperature=args.temperature,
    do_sample=True,
)


tf_cfg.gen_kwargs["pad_token_id"] = (
    tokenizer.pad_token_id
)


if terminators:

    if len(terminators) == 1:

        tf_cfg.gen_kwargs["eos_token_id"] = (
            terminators[0]
        )

    else:

        tf_cfg.gen_kwargs["eos_token_id"] = (
            terminators
        )


# ================= WATERMARK OBJECTS =================
kgw = KGW(
    KGWConfig(
        BASE / "config/KGW.json",
        tf_cfg,
    ),
    tf_cfg,
)

uni = Unigram(
    UnigramConfig(
        BASE / "config/Unigram.json",
        tf_cfg,
    ),
    tf_cfg,
)

sweet = SWEET(
    SWEETConfig(
        BASE / "config/SWEET.json",
        tf_cfg,
    ),
    tf_cfg,
)

ewd = EWD(
    EWDConfig(
        BASE / "config/EWD.json",
        tf_cfg,
    ),
    tf_cfg,
)

rankaware = RankAware(
    RankAwareConfig(
        BASE / "config/RankAware.json",
        tf_cfg,
    ),
    tf_cfg,
)


methods = {
    "kgw": kgw,
    "unigram": uni,
    "sweet": sweet,
    "ewd": ewd,
    "rankaware": rankaware,
}


# =====================================================
# HELPERS
# =====================================================


# ================= GET FIRST N TOKENS =================
def get_prompt_tokens(
    text: str,
    n_tokens: int,
) -> str:

    """
    Take the first N tokens from the source text.

    IMPORTANT:

    This RAW source passage is always saved in:

        "original"

    regardless of whether the model is OPT,
    Llama, Qwen, etc.
    """

    ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"][:n_tokens]

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
    )


# ================= INSTRUCTION =================
def build_messages(
    source_prompt: str,
):

    """
    Used ONLY when an instruction/chat model
    is automatically detected.
    """

    return [
        {
            "role": "user",

            "content": (
                "Continue the following passage naturally "
                "and coherently. "
                "Generate only the continuation. "
                "Do not repeat the passage and do not add "
                "commentary.\n\n"
                "Passage:\n"
                f"{source_prompt}"
            ),
        }
    ]


# =====================================================
# VANILLA INPUT
# =====================================================
def build_vanilla_input(
    source_prompt: str,
):

    

    # -----------------------------------------
    # INSTRUCTION MODEL
    # -----------------------------------------
    if IS_INSTRUCTION_MODEL:

        messages = build_messages(
            source_prompt
        )

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        return encoded.to(DEVICE)

    # -----------------------------------------
    # BASE MODEL / OPT
    # -----------------------------------------
    encoded = tokenizer(
        source_prompt,
        return_tensors="pt",
    )

    return {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }


# =====================================================
# MARKLLM WATERMARK PROMPT
# =====================================================
def build_markllm_prompt(
    source_prompt: str,
) -> str:

   

    # =========================================
    # BASE MODEL: OPT etc.
    # =========================================
    if not IS_INSTRUCTION_MODEL:

        return source_prompt

    # =========================================
    # INSTRUCTION MODEL
    # =========================================
    messages = build_messages(
        source_prompt
    )

    # -----------------------------------------
    # Reference token IDs produced by the
    # OFFICIAL chat template
    # -----------------------------------------
    reference_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # -----------------------------------------
    # Render chat template into text
    # -----------------------------------------
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    candidates = [
        rendered,
    ]

    
    if (
        tokenizer.bos_token is not None
        and rendered.startswith(
            tokenizer.bos_token
        )
    ):

        without_bos = rendered[
            len(tokenizer.bos_token):
        ]

        candidates.append(
            without_bos
        )

    
    decoded_reference = tokenizer.decode(
        reference_ids,
        skip_special_tokens=False,
    )

    if decoded_reference not in candidates:

        candidates.append(
            decoded_reference
        )

   
    for candidate in candidates:

        candidate_ids = tokenizer(
            candidate,
            add_special_tokens=True,
        )["input_ids"]

        if candidate_ids == reference_ids:

            return candidate

    raise RuntimeError(
        "\nCould not construct a MarkLLM prompt "
        "matching the tokenizer's official chat "
        "template.\n\n"
        f"Model: {MODEL_PATH}\n\n"
        "Generation stopped to avoid using an "
        "incorrect instruction format."
    )


def strip_plain_prompt(
    full_text: str,
    prompt: str,
) -> str:

    """
    Used for base models such as OPT.
    """

    if full_text.startswith(prompt):

        return full_text[
            len(prompt):
        ].lstrip()

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    )["input_ids"]

    decoded_prompt = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=True,
    )

    if full_text.startswith(
        decoded_prompt
    ):

        return full_text[
            len(decoded_prompt):
        ].lstrip()

    decoded_clean = decoded_prompt.rstrip()

    if full_text.startswith(
        decoded_clean
    ):

        return full_text[
            len(decoded_clean):
        ].lstrip()

    raise RuntimeError(
        "\nCould not separate plain-model "
        "continuation from prompt.\n"
    )


# =====================================================
# INSTRUCTION CONTINUATION EXTRACTION
# =====================================================
def strip_instruction_prompt(
    full_text: str,
    markllm_prompt: str,
) -> str:

    """
    Remove:
        chat template
        instruction
        source passage
        assistant-generation prefix

    Keep ONLY:
        newly generated assistant text
    """

    prefix_ids = tokenizer(
        markllm_prompt,
        add_special_tokens=True,
    )["input_ids"]

    decoded_prefix = tokenizer.batch_decode(
        [prefix_ids],
        skip_special_tokens=True,
    )[0]

    if full_text.startswith(
        decoded_prefix
    ):

        return full_text[
            len(decoded_prefix):
        ].strip()

    prefix_clean = decoded_prefix.rstrip()

    if full_text.startswith(
        prefix_clean
    ):

        return full_text[
            len(prefix_clean):
        ].strip()

    raise RuntimeError(
        "\nCould not safely separate the generated "
        "assistant continuation from the instruction "
        "prompt.\n\n"
        f"Decoded prefix:\n"
        f"{repr(decoded_prefix[:300])}\n\n"
        f"Generated text:\n"
        f"{repr(full_text[:300])}\n\n"
        "Stopping because the instruction must never "
        "be written into 'sampled'."
    )

def extract_watermarked_continuation(
    full_text: str,
    generation_prompt: str,
) -> str:

    if IS_INSTRUCTION_MODEL:

        return strip_instruction_prompt(
            full_text,
            generation_prompt,
        )

    return strip_plain_prompt(
        full_text,
        generation_prompt,
    )


# ================= SAVE =================
def save_json(
    path: Path,
    records: list,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =====================================================
# SHOW EXPERIMENT CONFIGURATION
# =====================================================
print("\n========================================")
print("GENERATION CONFIGURATION")
print("========================================")

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Mode:",
    (
        "INSTRUCTION"
        if IS_INSTRUCTION_MODEL
        else "PLAIN CONTINUATION"
    ),
)

print(
    "Prompt tokens:",
    args.prompt_tokens,
)

print(
    "Generated tokens:",
    args.max_new_tokens,
)

print(
    "Temperature:",
    args.temperature,
)

print(
    "Output directory:",
    OUT_DIR,
)

print("========================================")


# =====================================================
# SHOW FIRST EXAMPLE
# =====================================================
if len(data) > 0:

    example_text = extract_text(
        data[0]
    )

    example_prompt = get_prompt_tokens(
        example_text,
        args.prompt_tokens,
    )

    print("\n========================================")
    print("EXAMPLE RAW SOURCE PROMPT")
    print("========================================")

    print(example_prompt)

    print("\n========================================")

    if IS_INSTRUCTION_MODEL:

        print(
            "AUTOMATIC MODE = INSTRUCTION"
        )

        print("========================================")

        example_messages = build_messages(
            example_prompt
        )

        print(
            example_messages[0]["content"]
        )

    else:

        print(
            "AUTOMATIC MODE = PLAIN CONTINUATION"
        )

        print("========================================")

        print(
            "No instruction added."
        )

        print(
            "The raw source prompt is passed "
            "directly to the model."
        )

    print("\n========================================")
    print("JSON OUTPUT")
    print("========================================")

    print(
        '"original" = raw first source tokens'
    )

    print(
        '"sampled"  = generated continuation ONLY'
    )

    print("========================================")


# =====================================================
# VANILLA
# =====================================================
print("\n===== Running vanilla =====")

vanilla_out = (
    OUT_DIR
    / f"vanilla_n{len(data)}.json"
)

vanilla_records = []


for i, item in enumerate(
    tqdm(
        data,
        desc="vanilla",
    )
):

    text = extract_text(
        item
    )

   
    prompt = get_prompt_tokens(
        text,
        args.prompt_tokens,
    )

  
    enc = build_vanilla_input(
        prompt
    )

    input_length = (
        enc["input_ids"].shape[-1]
    )

    # -----------------------------------------
    # GENERATE NEW TOKENS
    # -----------------------------------------
    with torch.no_grad():

        out_ids = model.generate(
            **enc,
            **tf_cfg.gen_kwargs,
        )

    
    continuation_ids = out_ids[
        0,
        input_length:
    ]

    continuation = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True,
    ).strip()

    vanilla_records.append(
        {
            "id": i,

            # ---------------------------------
            # RAW FIRST 50 SOURCE TOKENS
            # ---------------------------------
            "original": prompt,

            # ---------------------------------
            # ONLY GENERATED CONTINUATION
            # ---------------------------------
            "sampled": continuation,
        }
    )


save_json(
    vanilla_out,
    vanilla_records,
)

print(
    "Saved:",
    vanilla_out,
)


# =====================================================
# WATERMARKED
# =====================================================
for name, wm in methods.items():

    print(
        f"\n===== Running {name} ====="
    )

    out_path = (
        OUT_DIR
        / f"{name}_n{len(data)}.json"
    )

    records = []

    for i, item in enumerate(
        tqdm(
            data,
            desc=name,
        )
    ):

        text = extract_text(
            item
        )

        # -------------------------------------
        # ALWAYS first 50 raw source tokens
        # -------------------------------------
        prompt = get_prompt_tokens(
            text,
            args.prompt_tokens,
        )

        
        generation_prompt = (
            build_markllm_prompt(
                prompt
            )
        )

        # -------------------------------------
        # WATERMARK GENERATION
        # -------------------------------------
        full_gen = (
            wm.generate_watermarked_text(
                generation_prompt
            )
        )

        
        continuation = (
            extract_watermarked_continuation(
                full_gen,
                generation_prompt,
            )
        )

        
        records.append(
            {
                "id": i,

                # RAW first 50 source tokens
                "original": prompt,

                # ONLY generated continuation
                "sampled": continuation,
            }
        )

    save_json(
        out_path,
        records,
    )

    print(
        "Saved:",
        out_path,
    )


# =====================================================
# FINISH
# =====================================================
print("\n========================================")
print("GENERATION DONE")
print("========================================")

print(
    "Model:",
    MODEL_TAG,
)

print(
    "Detected mode:",
    (
        "INSTRUCTION"
        if IS_INSTRUCTION_MODEL
        else "PLAIN CONTINUATION"
    ),
)

print(
    "original = first",
    args.prompt_tokens,
    "raw source tokens",
)

print(
    "sampled  = generated continuation only"
)

print(
    "generated tokens =",
    args.max_new_tokens,
)

print(
    "Detection should use sampled ONLY."
)

print("========================================")