# RankawareMark: Rank-Aware Watermarking for Large Language Models

This repository contains the implementation of "Rank-Aware", an entropy-guided, rank-aware text watermarking method for large language models (LLMs). Rank-Aware is the author's proposed method in this repository. The code also includes several baseline watermarking methods for comparison, including KGW, SWEET, Unigram, and EWD.

The baseline watermarking implementations are based on the "MarkLLM toolkit".

 Overview

Rank-Aware is designed to improve watermark detectability, robustness, and generated text quality by incorporating token-rank information and entropy-guided watermark embedding.

During generation, Rank-Aware evaluates the Shannon entropy of the next-token distribution and applies watermark bias when the entropy satisfies a predefined threshold. Eligible tokens are selected using a keyed greenlist construction process, and rank-dependent weights are assigned according to their positions in the language model's probability distribution.

During detection, the generated text is evaluated using the corresponding watermark detection mechanism to determine whether a watermark is present.

The framework aims to maintain high detection accuracy while minimizing distortion in the generated text.

 Main Files

The repository contains the following components:

```text
watermark/rankaware/rankaware.py    # Rank-Aware watermark implementation
watermark/rankaware/utils.py       # Rank-Aware utilities
config/RankAware.json              # Rank-Aware hyperparameters
watermark_generation.py            # Text generation
watermark_detection.py             # Watermark detection
requirements.txt                   # Python dependencies
```

Additional scripts are provided for robustness evaluation, perplexity calculation, hyperparameter analysis, and visualization.

The generation and detection scripts support the proposed Rank-Aware method and the baseline watermarking methods integrated into the repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/CuriousMind5/RankawareMark.git
cd RankawareMark
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Rank-Aware Configuration

The default Rank-Aware configuration is stored in `config/RankAware.json`:

```json
{
  "algorithm_name": "RankAware",
  "gamma": 0.25,
  "delta": 2.0,
  "r0": 60.0,
  "entropy_tau": 0.9,
  "hash_key": 15485863,
  "perm_method": "randperm",
  "decay_mode": "inv",
  "z_threshold": 4.0
}
```



 Basic Usage

Generate watermarked text:

```bash
python watermark_generation.py \
  --model_path /path/to/local/model \
  --input_json /path/to/input_dataset.json \
  --num_samples 500 \
  --prompt_tokens 50 \
  --max_new_tokens 200 \
  --temperature 0.7
```

Run watermark detection:

```bash
python watermark_detection.py
```

Additional evaluation scripts can be used to calculate detection performance, evaluate perplexity, analyze robustness, and perform parameter-sensitivity experiments.

**Important:** Some detection and evaluation scripts require modification of their local input, output, and model paths before execution.

## Baseline Watermarking Methods

The proposed Rank-Aware method is compared against the following watermarking baselines:

1. KGW
2. SWEET
3. Unigram
4. EWD

The baseline implementations are based on the open-source MarkLLM toolkit.

MarkLLM repository:

https://github.com/THU-BPM/MarkLLM

The baseline implementations are included for comparative evaluation. Rank-Aware is the proposed method implemented in this repository.

## Models Used

The experiments use both pretrained and instruction-tuned language models.

### Pretrained Language Models

1. OPT-2.7B
2. OPT-6.7B

### Instruction-Tuned Language Models

1. Llama-3-8B-Instruct
2. Qwen3-8B
3. Qwen2-7B

### Model Checkpoints

All models were downloaded and used locally during the experiments.

| Model | Pretrained checkpoint |
|---|---|
| OPT-2.7B | `facebook/opt-2.7b` |
| OPT-6.7B | `facebook/opt-6.7b` |
| Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Qwen2-7B | `Qwen/Qwen2-7B` |

Additional local models were used for paraphrasing, back-translation, and perplexity evaluation.

## Important Notes on Local Model Paths

**All models used in this project were downloaded and loaded locally.**

The repository does not distribute pretrained model weights.

Some scripts contain absolute local paths, such as:

```text
/home/sy/...
```

Before running experiments on another machine, users must replace these paths with the appropriate locations of their downloaded checkpoints.

For example:

```text
Original:
  /home/sy/models/OPT-2.7B

Custom:
  /path/to/local/OPT-2.7B
```

The following paths may need to be updated:

- Language model checkpoint directories
- Model tokenizer directories
- Input dataset locations
- Generated output directories
- Detection result directories
- Perplexity evaluation model paths
- Paraphrasing model paths
- Back-translation model paths

Users should inspect the relevant experiment scripts and configurations before execution.

## Dataset Preparation

The experimental evaluation includes five datasets:

1. C4 RealNewsLike
2. CNN/DailyMail
3. OpenGen
4. Wiki-CSAI
5. Reddit ELI5

These datasets are used to evaluate watermark detection performance across different text domains.

The experiments investigate both clean generation and watermark robustness under text transformation attacks.

Dataset files should be placed in the corresponding local dataset directory, and input paths must be adjusted according to the user's environment.

## Experimental Evaluation

The repository provides code for evaluating the proposed Rank-Aware method and comparing it against baseline watermarking approaches.

### Detection Performance

Watermark detectability is evaluated using:

- True Positive Rate at 5% False Positive Rate (TPR@5% FPR)
- F1-score at 5% FPR
- Area Under the Receiver Operating Characteristic Curve (AUROC)
- Watermark detection z-score

### Text Quality

Generated text quality is evaluated using perplexity (PPL).

Lower perplexity indicates better language-model likelihood under the evaluation model.

### Robustness Evaluation

The robustness experiments consider paraphrasing and back-translation attacks.

Paraphrasing attacks include:

- Pegasus
- DIPPER
- T5-Parrot

Back-translation attacks include:

- English–German–English (EN-DE-EN)
- English–French–English (EN-FR-EN)

### Parameter Sensitivity

The repository also includes experiments analyzing the effects of:

- Watermark strength (`delta`)
- Greenlist proportion (`gamma`)
- Entropy threshold (`entropy_tau`)
- Rank-decay scaling (`r0`)

These experiments examine the trade-off between watermark detection accuracy and generated text quality.

## Experimental Setup

The watermarking experiments compare Rank-Aware against all supported baseline methods under consistent model and dataset settings.

The default Rank-Aware configuration uses:

| Parameter | Default value |
|---|---|
| Gamma | 0.25 |
| Delta | 2.0 |
| Entropy threshold | 0.9 |
| Rank-decay parameter | 60.0 |
| Hash key | 15485863 |
| Detection z-threshold | 4.0 |

The experiments include clean watermark detection, robustness evaluations, generated-text quality comparisons, and parameter-sensitivity analyses.

## Reproducibility

To reproduce the experiments:

1. Clone the repository and install the required dependencies.
2. Download the pretrained language models and the evaluation models.
3. Update all local model paths.
4. Prepare the corresponding datasets.
5. Configure the desired watermarking algorithm.
6. Generate watermarked and unwatermarked text.
7. Run the watermark detection and evaluation scripts.
8. Calculate detection and text-quality metrics.

All pretrained language models were used locally. The repository does not include the downloaded model checkpoints.

## Citation

If you use this repository or the Rank-Aware watermarking implementation in your research, please cite the associated paper when its citation information becomes available.

## Acknowledgments

The baseline watermarking methods used for comparison are based on the **MarkLLM toolkit**:

https://github.com/THU-BPM/MarkLLM

We acknowledge the original authors and developers of the baseline watermarking algorithms, pretrained language models, and evaluation datasets used in this research.

# RankawareMark
