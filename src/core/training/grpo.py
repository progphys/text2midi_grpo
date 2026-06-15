from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import json
import math

import torch

from core.rewards import batch_rewards
from core.critics.metrics import build_default_metric_critics, score_symbolic_scores_with_critics
from core.score_selection import select_scores_for_reward

log = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    generated: torch.Tensor
    scores: list
    reward_scores: list
    track_selection_stats: list[dict]
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor
    log_prob_mask: torch.Tensor


def _compute_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    return ((rewards - mean) / std).view(-1)


class GRPOTrainer:
    def __init__(self, cfg, adapter, device: torch.device):
        self.cfg = cfg
        self.adapter = adapter
        self.device = device
        self.optimizer = torch.optim.AdamW(
            adapter.trainable_parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        observer_cfg = cfg.reward.get("observer_critic")
        score_metric_critics = bool(cfg.reward.get("score_metric_critics", False))
        if observer_cfg and observer_cfg.get("enabled", False):
            all_metric_critics = build_default_metric_critics(cfg.project_root)
        elif score_metric_critics:
            all_metric_critics = build_default_metric_critics(cfg.project_root)
        else:
            all_metric_critics = []
        self.metric_critics = all_metric_critics
        reward_checkpoint = None
        if observer_cfg:
            reward_checkpoint = Path(observer_cfg.checkpoint_path)
            if not reward_checkpoint.is_absolute():
                reward_checkpoint = Path(cfg.project_root) / reward_checkpoint
            reward_checkpoint = reward_checkpoint.resolve()
        self.reward_critic_name = None
        self.reward_critic_signal = "rank_score"
        if observer_cfg and observer_cfg.get("enabled", False):
            self.reward_critic_signal = str(observer_cfg.get("reward_signal", "rank_score"))
            for spec in all_metric_critics:
                if spec.client.checkpoint_path.resolve() == reward_checkpoint:
                    self.reward_critic_name = spec.name
                    self.metric_critics = [spec]
                    break
            if self.reward_critic_name is None:
                log.warning(
                    "Enabled observer reward checkpoint was not matched to a registered critic metric: %s",
                    reward_checkpoint,
                )

    def _lr_for_step(self, step: int) -> float:
        scheduler_cfg = self.cfg.training.get("lr_scheduler")
        base_lr = float(self.cfg.training.lr)
        if not scheduler_cfg or not bool(scheduler_cfg.get("enabled", False)):
            return base_lr

        kind = str(scheduler_cfg.get("type", "cosine"))
        warmup_steps = max(int(scheduler_cfg.get("warmup_steps", 0) or 0), 0)
        min_lr = float(scheduler_cfg.get("min_lr", 0.0) or 0.0)
        max_steps = max(int(self.cfg.training.max_steps), 1)

        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * float(step + 1) / float(warmup_steps)
        if kind != "cosine":
            return base_lr

        decay_start = warmup_steps
        decay_steps = max(max_steps - decay_start, 1)
        progress = min(max((step - decay_start) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    def set_lr_for_step(self, step: int) -> float:
        lr = self._lr_for_step(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _candidate_is_accepted(self, stats: dict, min_selected_tracks: int) -> bool:
        return int(stats.get("selected_track_count", 0) or 0) >= int(min_selected_tracks)

    def _log_track_selection(self, prompt_idx: int, attempt: int, stats: dict, accepted: bool) -> None:
        status = "ACCEPT_CANDIDATE" if accepted else "REJECT_CANDIDATE"
        log.info(
            "%s prompt=%02d attempt=%02d raw_tracks=%s candidate_tracks=%s selected_tracks=%s selected=%s",
            status,
            prompt_idx,
            attempt,
            stats.get("raw_track_count", 0),
            stats.get("candidate_track_count", 0),
            stats.get("selected_track_count", 0),
            stats.get("selected_track_names", []),
        )
        for item in stats.get("track_decisions", []):
            log.info(
                "  track=%02d %-6s reason=%s notes=%s rank=%s name=%s",
                int(item.get("track_index", -1)),
                str(item.get("decision", "")).upper(),
                item.get("reason"),
                item.get("note_count"),
                item.get("rank_score"),
                item.get("track_name"),
            )

    def _generate_with_track_rejection(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        selection_cfg = self.cfg.reward.get("track_selection", {}) or {}
        group_size = int(self.cfg.grpo.num_rollouts)
        max_attempts = max(int(selection_cfg.get("max_attempts_per_prompt", group_size * 4)), group_size)
        min_selected_tracks = max(int(selection_cfg.get("min_selected_tracks", 1)), 1)
        proposal_batch_size = max(int(selection_cfg.get("proposal_batch_size", group_size)), 1)
        max_tracks = int(selection_cfg.get("max_tracks", self.cfg.reward.get("max_tracks", 4)))
        drop_drums = bool(selection_cfg.get("drop_drums", True))

        accepted_chunks: list[torch.Tensor] = []
        for prompt_idx in range(input_ids.shape[0]):
            accepted_for_prompt: list[torch.Tensor] = []
            rejected_pool: list[tuple[int, torch.Tensor, dict]] = []
            attempt = 0
            while len(accepted_for_prompt) < group_size and attempt < max_attempts:
                remaining_attempts = max_attempts - attempt
                proposal_count = min(proposal_batch_size, remaining_attempts)
                candidate_batch = self.adapter.generate_repeated(
                    input_ids=input_ids[prompt_idx : prompt_idx + 1],
                    attention_mask=attention_mask[prompt_idx : prompt_idx + 1],
                    num_return_sequences=proposal_count,
                    max_len=self.cfg.grpo.rollout_max_len,
                    temperature=self.cfg.grpo.rollout_temperature,
                )
                candidate_scores = self.adapter.decode_scores(candidate_batch)
                _, stats_list = select_scores_for_reward(
                    candidate_scores,
                    max_tracks=max_tracks,
                    drop_drums=drop_drums,
                )
                for candidate_idx, stats in enumerate(stats_list):
                    attempt += 1
                    candidate = candidate_batch[candidate_idx : candidate_idx + 1]
                    accepted = self._candidate_is_accepted(stats, min_selected_tracks=min_selected_tracks)
                    self._log_track_selection(prompt_idx, attempt, stats, accepted)
                    if accepted:
                        accepted_for_prompt.append(candidate)
                    else:
                        rejected_pool.append((int(stats.get("selected_track_count", 0) or 0), candidate, stats))
                    if len(accepted_for_prompt) >= group_size or attempt >= max_attempts:
                        break

            while len(accepted_for_prompt) < group_size:
                if not rejected_pool:
                    log.warning(
                        "No fallback candidates available for prompt=%02d after %d attempts; generating one without rejection.",
                        prompt_idx,
                        attempt,
                    )
                    fallback = self.adapter.generate_repeated(
                        input_ids=input_ids[prompt_idx : prompt_idx + 1],
                        attention_mask=attention_mask[prompt_idx : prompt_idx + 1],
                        num_return_sequences=1,
                        max_len=self.cfg.grpo.rollout_max_len,
                        temperature=self.cfg.grpo.rollout_temperature,
                    )
                    accepted_for_prompt.append(fallback)
                    continue
                rejected_pool.sort(key=lambda item: item[0], reverse=True)
                selected_count, fallback, stats = rejected_pool.pop(0)
                log.warning(
                    "FALLBACK_CANDIDATE prompt=%02d selected_tracks=%d min_required=%d selected=%s",
                    prompt_idx,
                    selected_count,
                    min_selected_tracks,
                    stats.get("selected_track_names", []),
                )
                accepted_for_prompt.append(fallback)

            accepted_chunks.extend(accepted_for_prompt)
        return torch.cat(accepted_chunks, dim=0)

    @torch.no_grad()
    def rollout(self, captions: list[str]) -> tuple[RolloutBatch, list[dict[str, float]]]:
        group_size = self.cfg.grpo.num_rollouts
        log.info(
            "Rollout started: prompts=%d num_rollouts=%d total_sequences=%d rollout_max_len=%d generation_chunk_size=%d",
            len(captions),
            group_size,
            len(captions) * group_size,
            int(self.cfg.grpo.rollout_max_len),
            int(self.cfg.grpo.get("generation_chunk_size", 1)),
        )
        input_ids, attention_mask = self.adapter.tokenize_captions(captions)
        selection_cfg = self.cfg.reward.get("track_selection", {}) or {}
        rejection_enabled = bool(selection_cfg.get("rejection_sampling", {}).get("enabled", False))
        if rejection_enabled:
            generated = self._generate_with_track_rejection(input_ids, attention_mask)
        else:
            generated = self.adapter.generate_repeated(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=group_size,
                max_len=self.cfg.grpo.rollout_max_len,
                temperature=self.cfg.grpo.rollout_temperature,
            )
        log.info("Rollout generation complete: generated_shape=%s", tuple(generated.shape))

        input_ids_rep = input_ids.repeat_interleave(group_size, dim=0)
        attention_mask_rep = attention_mask.repeat_interleave(group_size, dim=0)
        old_log_probs, log_prob_mask = self.adapter.score_sequence_tokens(
            input_ids_rep, attention_mask_rep, generated, use_reference=False
        )

        scores = self.adapter.decode_scores(generated)
        if bool(selection_cfg.get("enabled", False)):
            reward_scores, track_selection_stats = select_scores_for_reward(
                scores,
                max_tracks=int(selection_cfg.get("max_tracks", self.cfg.reward.get("max_tracks", 4))),
                drop_drums=bool(selection_cfg.get("drop_drums", True)),
            )
        else:
            reward_scores = scores
            track_selection_stats = [
                {
                    "raw_track_count": len([track for track in getattr(score, "tracks", []) or [] if getattr(track, "notes", None)])
                    if score is not None else 0,
                    "candidate_track_count": 0,
                    "selected_track_count": 0,
                    "selected_track_indices": [],
                    "selected_track_names": [],
                }
                for score in scores
            ]
        repeated_captions = [caption for caption in captions for _ in range(group_size)]
        reward_dicts = batch_rewards(reward_scores, self.cfg.reward, captions=repeated_captions)
        critic_metrics = score_symbolic_scores_with_critics(
            project_root=self.cfg.project_root,
            critic_specs=self.metric_critics,
            scores=reward_scores,
            captions=repeated_captions,
            group_size=group_size,
            tmp_prefix="observer_reward_",
        )
        for reward, critic_values, selection_stats in zip(reward_dicts, critic_metrics, track_selection_stats):
            reward["raw_track_count"] = float(selection_stats.get("raw_track_count", 0) or 0)
            reward["selected_track_count"] = float(selection_stats.get("selected_track_count", 0) or 0)
            reward.update(critic_values)
            observer_metric_name = self.reward_critic_name or "observer_default"
            reward["observer"] = float(critic_values.get(observer_metric_name, 0.0) or 0.0)
            reward["observer_raw"] = float(critic_values.get(f"{observer_metric_name}_raw", 0.0) or 0.0)
            reward["observer_error"] = critic_values.get(f"{observer_metric_name}_error")
            invalid_blocks_reward = bool(self.cfg.reward.get("invalid_enabled", True)) and reward["invalid"] > 0.0
            if self.reward_critic_name and not invalid_blocks_reward:
                reward["total"] += self.cfg.reward.observer_weight * self._observer_reward_signal(critic_values)
        rewards = torch.tensor(
            [reward["total"] for reward in reward_dicts],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = _compute_advantages(rewards, group_size)

        return (
            RolloutBatch(
                input_ids=input_ids_rep,
                attention_mask=attention_mask_rep,
                generated=generated,
                scores=scores,
                reward_scores=reward_scores,
                track_selection_stats=track_selection_stats,
                rewards=rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                log_prob_mask=log_prob_mask,
            ),
            reward_dicts,
        )

    def _observer_reward_signal(self, critic_values: dict[str, float | str | None]) -> float:
        metric_name = self.reward_critic_name
        if not metric_name:
            return 0.0
        if self.reward_critic_signal == "pairwise_win_rate":
            return float(critic_values.get(f"{metric_name}_pairwise_win_rate", 0.0) or 0.0)
        return float(critic_values.get(f"{metric_name}_rank_score", 0.0) or 0.0)

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        self.adapter.model.train()

        eps = self.cfg.grpo.epsilon
        beta = self.cfg.grpo.beta
        mini_batch = self.cfg.grpo.update_mini_batch
        n_sequences = batch.generated.shape[0]

        ref_lp_chunks = []
        mask_chunks = []
        with torch.no_grad():
            for start in range(0, n_sequences, mini_batch):
                stop = start + mini_batch
                ref_lp, ref_mask = self.adapter.score_sequence_tokens(
                    batch.input_ids[start:stop],
                    batch.attention_mask[start:stop],
                    batch.generated[start:stop],
                    use_reference=True,
                )
                ref_lp_chunks.append(ref_lp)
                mask_chunks.append(ref_mask)
        ref_lp = torch.cat(ref_lp_chunks)
        ref_mask = torch.cat(mask_chunks)

        with torch.no_grad():
            pre_new_lp, pre_mask = self.adapter.score_sequence_tokens(
                batch.input_ids[:mini_batch],
                batch.attention_mask[:mini_batch],
                batch.generated[:mini_batch],
                use_reference=False,
            )
            pre_kl_tensor = self._token_kl(pre_new_lp, ref_lp[:mini_batch]) * pre_mask
            pre_kl = (pre_kl_tensor.sum() / pre_mask.sum().clamp(min=1.0)).item()

        kl_ceiling = self.cfg.grpo.get("kl_ceiling")
        if kl_ceiling is not None and pre_kl > float(kl_ceiling):
            log.warning(
                "KL=%.3f > %.3f, skipping update step",
                pre_kl,
                float(kl_ceiling),
            )
            return {
                "loss": 0.0,
                "kl": pre_kl,
                "grad_norm": 0.0,
                "mean_ratio": 1.0,
                "skipped": 1.0,
            }

        if pre_kl < self.cfg.grpo.kl_floor:
            log.warning(
                "KL=%.3f < %.3f, skipping update step",
                pre_kl,
                self.cfg.grpo.kl_floor,
            )
            return {
                "loss": 0.0,
                "kl": pre_kl,
                "grad_norm": 0.0,
                "mean_ratio": 1.0,
                "skipped": 1.0,
            }

        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        total_ratio = torch.tensor(0.0, device=self.device)
        total_tokens = torch.tensor(0.0, device=self.device)
        total_valid_tokens = (batch.log_prob_mask.detach() * ref_mask).sum().clamp(min=1.0)

        for start in range(0, n_sequences, mini_batch):
            stop = start + mini_batch
            new_lp, token_mask = self.adapter.score_sequence_tokens(
                batch.input_ids[start:stop],
                batch.attention_mask[start:stop],
                batch.generated[start:stop],
                use_reference=False,
            )
            old_lp = batch.old_log_probs[start:stop].detach()
            advantages = batch.advantages[start:stop].detach()
            ref_chunk = ref_lp[start:stop]
            token_mask = token_mask * batch.log_prob_mask[start:stop].detach() * ref_mask[start:stop]

            ratio = torch.exp(new_lp - old_lp)
            advantages = advantages.unsqueeze(-1)
            surrogate = torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - eps, 1 + eps) * advantages,
            )
            kl_chunk = self._token_kl(new_lp, ref_chunk)
            chunk_loss = ((-surrogate + beta * kl_chunk) * token_mask).sum() / total_valid_tokens
            chunk_loss.backward()

            total_loss += ((-surrogate + beta * kl_chunk) * token_mask).sum().detach()
            total_kl += (kl_chunk * token_mask).sum().detach()
            total_ratio += (ratio * token_mask).sum().detach()
            total_tokens += token_mask.sum().detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.adapter.trainable_parameters(),
            max_norm=self.cfg.training.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss": (total_loss / total_tokens.clamp(min=1.0)).item(),
            "kl": (total_kl / total_tokens.clamp(min=1.0)).item(),
            "grad_norm": float(grad_norm),
            "mean_ratio": (total_ratio / total_tokens.clamp(min=1.0)).item(),
            "skipped": 0.0,
        }

    @staticmethod
    def _token_kl(policy_log_probs: torch.Tensor, reference_log_probs: torch.Tensor) -> torch.Tensor:
        log_ratio_ref = reference_log_probs - policy_log_probs
        return torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    def save(self, step: int) -> None:
        from omegaconf import OmegaConf

        output_dir = os.path.join(self.cfg.paths.checkpoint_dir, f"step_{step:05d}")
        os.makedirs(output_dir, exist_ok=True)
        self.adapter.save_adapter(output_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        Path(os.path.join(output_dir, "trainer_state.json")).write_text(
            json.dumps({"global_step": int(step)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        OmegaConf.save(self.cfg, os.path.join(output_dir, "config.yaml"))
        log.info("Checkpoint saved to %s", output_dir)

    def load_state(self, checkpoint_dir: str | os.PathLike[str]) -> int:
        checkpoint_dir = str(checkpoint_dir)
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        global_step = 0

        if os.path.exists(optimizer_path):
            try:
                state = torch.load(optimizer_path, map_location=self.device, weights_only=False)
                self.optimizer.load_state_dict(state)
            except Exception:
                log.exception("Failed to load optimizer state from %s", optimizer_path)

        if os.path.exists(trainer_state_path):
            try:
                payload = json.loads(Path(trainer_state_path).read_text(encoding="utf-8"))
                global_step = int(payload.get("global_step", 0) or 0)
            except Exception:
                log.exception("Failed to load trainer state from %s", trainer_state_path)

        return global_step
