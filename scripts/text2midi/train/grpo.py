"""
GRPO trainer for text2midi.

Algorithm (Group Relative Policy Optimization):
  1. Sample G MIDI sequences per caption (rollout, no grad)
  2. Compute reward for each sequence
  3. Normalize rewards within the group → advantages
  4. Forward pass with grad to get log-probs
  5. PPO-clip loss + KL penalty → update LoRA weights only

Key design decisions:
  - Generation is batched: G captions (same text repeated G times) → one model.generate call
  - Reference model is the frozen base (before LoRA) for KL computation
  - Only LoRA parameters receive gradients
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import T5Tokenizer
from peft import LoraConfig, get_peft_model

log = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    """One batch of generated sequences with their rewards."""
    input_ids: torch.Tensor          # [B, src_len]
    attention_mask: torch.Tensor     # [B, src_len]
    generated: torch.Tensor          # [B*G, tgt_len]
    rewards: torch.Tensor            # [B*G]
    advantages: torch.Tensor         # [B*G]
    old_log_probs: torch.Tensor      # [B*G]  — from generation step


def _build_lora_config(cfg: DictConfig) -> LoraConfig:
    kwargs = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
    )
    layers = cfg.lora.get("layers_to_transform")
    if layers is not None:
        kwargs["layers_to_transform"] = list(layers)
        # tells PEFT where to find the layer index in the module path
        # e.g. "decoder.layers.12.self_attn.to_qkv" → pattern "layers"
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _sequence_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generated: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean per-token log-prob for each sequence (not sum).

    Normalising by sequence length makes the KL threshold
    interpretable regardless of rollout_max_len.

    generated: [N, tgt_len]  — token ids including start token
    Returns: [N] — one scalar per sequence (mean over non-pad tokens)
    """
    tgt_in  = generated[:, :-1]   # [N, tgt_len-1]
    tgt_out = generated[:, 1:]    # [N, tgt_len-1]

    logits    = model(input_ids, attention_mask, tgt_in)
    log_probs = F.log_softmax(logits, dim=-1)

    token_lp = log_probs.gather(
        dim=-1,
        index=tgt_out.unsqueeze(-1),
    ).squeeze(-1)   # [N, tgt_len-1]

    # NaN guard — заменяем nan/inf нулём перед суммированием
    token_lp = torch.nan_to_num(token_lp, nan=0.0, posinf=0.0, neginf=0.0)

    mask    = (tgt_out != 0).float()
    lengths = mask.sum(dim=-1).clamp(min=1)
    seq_lp  = (token_lp * mask).sum(dim=-1) / lengths  # mean per token [N]
    return seq_lp


def _compute_advantages(rewards: torch.Tensor, G: int) -> torch.Tensor:
    """
    Normalize rewards within each group of G.

    rewards: [B*G]
    Returns: [B*G]
    """
    rewards = rewards.view(-1, G)               # [B, G]
    mean = rewards.mean(dim=1, keepdim=True)    # [B, 1]
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards - mean) / std         # [B, G]
    return advantages.view(-1)                  # [B*G]


class GRPOTrainer:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        repo = cfg.paths.model_repo
        sys.path.insert(0, repo)

        from model.transformer_model import Transformer
        from omegaconf import OmegaConf

        repo_cfg = OmegaConf.load(os.path.join(repo, "configs/config.yaml"))
        m = repo_cfg.model.text2midi_model

        # Load REMI tokenizer
        with open(os.path.join(cfg.paths.weights_dir, "vocab_remi.pkl"), "rb") as f:
            self.r_tokenizer = pickle.load(f)
        vocab_size = len(self.r_tokenizer)

        # Load T5 tokenizer
        self.t5_tokenizer = T5Tokenizer.from_pretrained(cfg.paths.tokenizer_dir)

        # Base model (trainable with LoRA)
        base = Transformer(
            n_vocab=vocab_size,
            d_model=m.decoder_d_model,
            nhead=m.decoder_num_heads,
            max_len=m.decoder_max_sequence_length,
            num_decoder_layers=m.decoder_num_layers,
            dim_feedforward=m.decoder_intermediate_size,
            use_moe=m.use_moe,
            device=device,
        )
        base.load_state_dict(torch.load(
            os.path.join(cfg.paths.weights_dir, "pytorch_model.bin"),
            map_location=device,
            weights_only=False,
        ))

        lora_cfg = _build_lora_config(cfg)
        self.model = get_peft_model(base, lora_cfg).to(device)
        self.model.print_trainable_parameters()

        # Reference model — frozen original weights for KL
        self.ref_model = Transformer(
            n_vocab=vocab_size,
            d_model=m.decoder_d_model,
            nhead=m.decoder_num_heads,
            max_len=m.decoder_max_sequence_length,
            num_decoder_layers=m.decoder_num_layers,
            dim_feedforward=m.decoder_intermediate_size,
            use_moe=m.use_moe,
            device=device,
        )
        self.ref_model.load_state_dict(torch.load(
            os.path.join(cfg.paths.weights_dir, "pytorch_model.bin"),
            map_location=device,
            weights_only=False,
        ))
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rollout
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def rollout(self, captions: list[str]) -> RolloutBatch:
        """
        Generate G sequences for each caption, compute rewards and advantages.

        captions: list of B strings
        Returns: RolloutBatch
        """
        from rewards import batch_rewards

        G = self.cfg.grpo.num_rollouts
        B = len(captions)

        # Tokenize captions, repeat each G times → [B*G, src_len]
        enc = self.t5_tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc.input_ids.to(self.device)          # [B, src_len]
        attention_mask = enc.attention_mask.to(self.device)

        input_ids_rep = input_ids.repeat_interleave(G, dim=0)       # [B*G, src_len]
        attention_mask_rep = attention_mask.repeat_interleave(G, dim=0)

        # Generate one sequence at a time — the original model.generate() has a bug
        # with batch_size > 1: it always takes `next_tokens[-1]` from a flat tensor,
        # which gives only the last token of the last sequence in the batch.
        self.model.eval()
        seqs = []
        pad = torch.ones(self.cfg.grpo.rollout_max_len, dtype=torch.long, device=self.device)
        for i in range(B * G):
            try:
                out = self.model.generate(
                    input_ids_rep[i : i + 1],
                    attention_mask_rep[i : i + 1],
                    max_len=self.cfg.grpo.rollout_max_len,
                    temperature=self.cfg.grpo.rollout_temperature,
                )   # [1, tgt_len]
                seq = out[0]
                if torch.isnan(seq.float()).any() or torch.isinf(seq.float()).any():
                    seq = pad
            except Exception:
                seq = pad
            seqs.append(seq)
        generated = torch.stack(seqs, dim=0)  # [B*G, tgt_len]

        # Old log-probs (for ratio in GRPO loss)
        old_lp = _sequence_log_probs(
            self.model, input_ids_rep, attention_mask_rep, generated
        )   # [B*G]

        # Decode to MIDI scores for reward computation
        scores = []
        for i in range(B * G):
            try:
                score = self.r_tokenizer.decode(generated[i].tolist())
            except Exception:
                score = None
            scores.append(score)

        reward_dicts = batch_rewards(scores, self.cfg.reward)
        rewards_tensor = torch.tensor(
            [d["total"] for d in reward_dicts],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = _compute_advantages(rewards_tensor, G)

        return RolloutBatch(
            input_ids=input_ids_rep,
            attention_mask=attention_mask_rep,
            generated=generated,
            rewards=rewards_tensor,
            advantages=advantages,
            old_log_probs=old_lp,
        ), reward_dicts

    # ─────────────────────────────────────────────────────────────────────────
    # GRPO update step
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """One GRPO gradient step with mini-batch forward to avoid OOM."""
        self.model.train()

        eps  = self.cfg.grpo.epsilon
        beta = self.cfg.grpo.beta
        mb   = self.cfg.grpo.update_mini_batch  # sequences per forward pass

        N = batch.generated.shape[0]  # B*G

        # ── ref log-probs (no grad, also mini-batched) ────────────────────────
        ref_lp_list = []
        with torch.no_grad():
            for i in range(0, N, mb):
                ref_lp_list.append(_sequence_log_probs(
                    self.ref_model,
                    batch.input_ids[i:i+mb],
                    batch.attention_mask[i:i+mb],
                    batch.generated[i:i+mb],
                ))
        ref_lp = torch.cat(ref_lp_list)

        # ── KL guard: проверяем до обновления ─────────────────────────────────
        # Если модель уже сильно дивергировала — пропускаем шаг
        with torch.no_grad():
            pre_kl = (_sequence_log_probs(
                self.model, batch.input_ids[:mb],
                batch.attention_mask[:mb], batch.generated[:mb],
            ) - ref_lp[:mb]).mean().item()

        # Per-token KL threshold: -0.5 означает что в среднем каждый токен
        # на 0.5 log-prob единицы менее вероятен чем у референса — это уже много
        KL_THRESHOLD = -0.5
        if pre_kl < KL_THRESHOLD:
            log.warning(f"KL={pre_kl:.3f} < {KL_THRESHOLD} — пропускаем шаг обновления")
            return {"loss": 0.0, "kl": pre_kl, "grad_norm": 0.0, "mean_ratio": 1.0,
                    "skipped": 1.0}

        # ── policy update: accumulate loss over mini-batches ──────────────────
        self.optimizer.zero_grad()

        total_loss  = torch.tensor(0.0, device=self.device)
        total_kl    = torch.tensor(0.0, device=self.device)
        total_ratio = torch.tensor(0.0, device=self.device)

        for i in range(0, N, mb):
            new_lp = _sequence_log_probs(
                self.model,
                batch.input_ids[i:i+mb],
                batch.attention_mask[i:i+mb],
                batch.generated[i:i+mb],
            )
            old_lp = batch.old_log_probs[i:i+mb].detach()
            A      = batch.advantages[i:i+mb].detach()
            r_lp   = ref_lp[i:i+mb]

            ratio     = torch.exp(new_lp - old_lp)
            surrogate = torch.min(ratio * A, torch.clamp(ratio, 1-eps, 1+eps) * A)
            kl_chunk  = (new_lp - r_lp)

            chunk_loss = (-surrogate.sum() + beta * kl_chunk.sum()) / N
            chunk_loss.backward()

            total_loss  = total_loss  + (-surrogate.sum().detach() + beta * kl_chunk.sum().detach())
            total_kl    = total_kl    + kl_chunk.sum().detach()
            total_ratio = total_ratio + ratio.sum().detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            max_norm=self.cfg.training.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss":       (total_loss  / N).item(),
            "kl":         (total_kl    / N).item(),
            "grad_norm":  grad_norm.item(),
            "mean_ratio": (total_ratio / N).item(),
            "skipped":    0.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, step: int):
        from omegaconf import OmegaConf
        out = os.path.join(self.cfg.paths.checkpoint_dir, f"step_{step:05d}")
        os.makedirs(out, exist_ok=True)
        self.model.save_pretrained(out)
        OmegaConf.save(self.cfg, os.path.join(out, "config.yaml"))
        log.info(f"Checkpoint saved to {out}")
