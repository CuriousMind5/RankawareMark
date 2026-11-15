
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import torch
from functools import partial
from transformers import LogitsProcessor, LogitsProcessorList

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig

from .utils import hash64, green_mask_from_seed


# ============================================================
# Config
# ============================================================

class RankAwareConfig(BaseConfig):

    def initialize_parameters(self) -> None:
        self.gamma = float(self.config_dict["gamma"])
        self.delta = float(self.config_dict["delta"])
        self.r0 = float(self.config_dict["r0"])
        self.entropy_tau = float(self.config_dict["entropy_tau"])
        self.hash_key = int(self.config_dict["hash_key"])

        self.perm_method = str(self.config_dict.get("perm_method", "randperm"))
        self.decay_mode = str(self.config_dict["decay_mode"])
        self.z_threshold = float(self.config_dict["z_threshold"])

    @property
    def algorithm_name(self) -> str:
        return "RankAware"


# ============================================================
# Utils
# ============================================================

class RankAwareUtils:

    def __init__(self, config: RankAwareConfig):
        self.cfg = config
        self.V = int(config.vocab_size)
        self.m = int(round(config.gamma * self.V))

    @torch.no_grad()
    def _weights(self, ranks: torch.Tensor) -> torch.Tensor:
        r = ranks.float()

        if self.cfg.decay_mode == "exp":
            w = torch.exp(-r / self.cfg.r0)
        else:
            w = 1.0 / (1.0 + (r / self.cfg.r0))

        return torch.clamp(w, min=1e-8)

    # ---------- GENERATION BIAS ----------

    @torch.no_grad()
    def apply_rankaware_bias(self, input_ids, scores):
        B, V = scores.shape
        device = scores.device

        for b in range(B):

            probs = torch.softmax(scores[b].float(), dim=-1)
            H = -torch.sum(probs * torch.log(probs + 1e-12)).item()

            # Entropy gate:
            # If entropy is low, skip watermark bias.
            if H < self.cfg.entropy_tau:
                continue

            prev = int(input_ids[b, -1].item())
            seed = hash64(prev ^ self.cfg.hash_key)

            green = green_mask_from_seed(
                seed,
                self.V,
                self.m,
                device,
                self.cfg.perm_method,
            )

            order = torch.argsort(scores[b], descending=True)
            ranks = torch.empty(self.V, device=device, dtype=torch.long)
            ranks[order] = torch.arange(self.V, device=device)

            w = self._weights(ranks)
            wg = w[green]
            s = wg.sum()
            norm = (self.m / s) if s.item() > 0 else 1.0

            scores[b][green] += self.cfg.delta * norm * wg

        return scores

    # ---------- DETECTION SCORE ----------

    @torch.no_grad()
    def score_sequence(self, model, ids: torch.Tensor):

        device = ids.device
        T = ids.numel()

        if T <= 1:
            return float("nan"), 0, 0

        logits = model(ids.unsqueeze(0)).logits[0]

        Hcnt = 0
        N = 0

        prev = int(ids[0].item())

        for t in range(1, T):

            probs = torch.softmax(logits[t - 1].float(), dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-12)).item()

            cur = int(ids[t].item())

            # Same entropy gate during detection.
            if ent < self.cfg.entropy_tau:
                prev = cur
                continue

            seed = hash64(prev ^ self.cfg.hash_key)

            green = green_mask_from_seed(
                seed,
                self.V,
                self.m,
                device,
                self.cfg.perm_method,
            )

            N += 1

            if green[cur].item():
                Hcnt += 1

            prev = cur

        if N <= 0:
            return float("nan"), 0, 0

        var = N * self.cfg.gamma * (1 - self.cfg.gamma)
        z = (Hcnt - N * self.cfg.gamma) / math.sqrt(var)

        return float(z), Hcnt, N


# ============================================================
# Logits Processor
# ============================================================

class RankAwareLogitsProcessor(LogitsProcessor):

    def __init__(self, config: RankAwareConfig, utils: RankAwareUtils):
        self.config = config
        self.utils = utils

    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, neginf=-1e9, posinf=1e9)
        return self.utils.apply_rankaware_bias(input_ids, scores)


# ============================================================
# Main Watermark Class
# ============================================================

class RankAware(BaseWatermark):

    def __init__(
        self,
        algorithm_config: str | RankAwareConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ):
        if isinstance(algorithm_config, str):
            self.config = RankAwareConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, RankAwareConfig):
            self.config = algorithm_config
        else:
            raise TypeError("Invalid algorithm_config type")

        self.utils = RankAwareUtils(self.config)
        self.logits_processor = RankAwareLogitsProcessor(self.config, self.utils)

    # ---------- GENERATION ----------

    def generate_watermarked_text(self, prompt: str) -> str:

        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )

        encoded_prompt = self.config.generation_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.config.device)

        encoded_output = generate_with_watermark(**encoded_prompt)

        text = self.config.generation_tokenizer.batch_decode(
            encoded_output,
            skip_special_tokens=True,
        )[0]

        return text

    # ---------- DETECTION ----------

    def detect_watermark(self, text: str, return_dict: bool = True):

        tok = self.config.generation_tokenizer
        model = self.config.generation_model
        device = self.config.device

        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"][0].to(device)

        z, H, N = self.utils.score_sequence(model, ids)

        flagged = (z >= self.config.z_threshold) if math.isfinite(z) else False

        if return_dict:
            return {
                "is_watermarked": bool(flagged),
                "score": float(z),
                "green_hits": int(H),
                "tokens_scored": int(N),
            }

        return bool(flagged), float(z)

