from __future__ import annotations

import logging

from core.critics.metrics import build_default_metric_critics, score_symbolic_scores_with_critics
from core.evaluation.text2midi_metrics import summarize_text2midi_metrics
from core.rewards import batch_rewards

log = logging.getLogger(__name__)


def _critic_group_summary(
    critic_metrics: list[dict[str, float | str | None]],
    metric_name: str,
    group_size: int,
) -> dict[str, float]:
    top1_raw_values: list[float] = []
    top1_gap_values: list[float] = []
    for group_start in range(0, len(critic_metrics), group_size):
        group = critic_metrics[group_start : group_start + group_size]
        valid_raw = [
            float(item.get(f"{metric_name}_raw", 0.0) or 0.0)
            for item in group
            if not item.get(f"{metric_name}_error")
        ]
        if not valid_raw:
            continue
        valid_raw.sort(reverse=True)
        top1_raw_values.append(valid_raw[0])
        if len(valid_raw) >= 2:
            top1_gap_values.append(valid_raw[0] - valid_raw[1])
        else:
            top1_gap_values.append(0.0)
    return {
        f"eval/{metric_name}_top1_raw": sum(top1_raw_values) / len(top1_raw_values) if top1_raw_values else 0.0,
        f"eval/{metric_name}_top1_gap": sum(top1_gap_values) / len(top1_gap_values) if top1_gap_values else 0.0,
    }


def _critic_mean_summary(
    critic_metrics: list[dict[str, float | str | None]],
    metric_name: str,
) -> dict[str, float]:
    n = max(len(critic_metrics), 1)
    failures = [1.0 if item.get(f"{metric_name}_error") else 0.0 for item in critic_metrics]
    return {
        f"eval/{metric_name}": sum(float(item.get(metric_name, 0.0) or 0.0) for item in critic_metrics) / n,
        f"eval/{metric_name}_raw": sum(float(item.get(f"{metric_name}_raw", 0.0) or 0.0) for item in critic_metrics) / n,
        f"eval/{metric_name}_rank": sum(float(item.get(f"{metric_name}_rank_score", 0.0) or 0.0) for item in critic_metrics) / n,
        f"eval/{metric_name}_pairwise": sum(float(item.get(f"{metric_name}_pairwise_win_rate", 0.0) or 0.0) for item in critic_metrics) / n,
        f"eval/{metric_name}_success_rate": 1.0 - sum(failures) / n,
    }


def evaluate_prompts(
    adapter,
    prompts: list[str],
    generations_per_prompt: int,
    max_len: int,
    temperature: float,
    reward_cfg,
) -> dict[str, float]:
    all_rewards = []
    all_critic_metrics = []
    all_decoded_scores = []
    all_captions = []
    critic_specs = build_default_metric_critics(adapter.cfg.project_root)

    for idx, caption in enumerate(prompts, start=1):
        input_ids, attention_mask = adapter.tokenize_captions([caption])
        generated = adapter.generate_repeated(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=generations_per_prompt,
            max_len=max_len,
            temperature=temperature,
        )

        decoded_scores = adapter.decode_scores(generated)
        prompt_captions = [caption] * len(decoded_scores)
        all_decoded_scores.extend(decoded_scores)
        all_captions.extend(prompt_captions)
        all_critic_metrics.extend(
            score_symbolic_scores_with_critics(
                project_root=adapter.cfg.project_root,
                critic_specs=critic_specs,
                scores=decoded_scores,
                captions=prompt_captions,
                group_size=generations_per_prompt,
                tmp_prefix="observer_eval_",
            )
        )

        all_rewards.extend(batch_rewards(decoded_scores, reward_cfg, captions=prompt_captions))

        if idx % 10 == 0:
            valid = sum(1 for reward in all_rewards if reward["invalid"] == 0)
            log.info(
                "  %d/%d prompts evaluated, valid so far: %d/%d",
                idx,
                len(prompts),
                valid,
                len(all_rewards),
            )

    n = len(all_rewards)
    summary = {
        "eval/reward_total": sum(item["total"] for item in all_rewards) / n,
        "eval/reward_key": sum(item["key"] for item in all_rewards) / n,
        "eval/reward_key_conditioned": sum(item["key_conditioned"] for item in all_rewards) / n,
        "eval/reward_rhythm": sum(item["rhythm"] for item in all_rewards) / n,
        "eval/reward_meter": sum(item["meter"] for item in all_rewards) / n,
        "eval/reward_tempo": sum(item["tempo"] for item in all_rewards) / n,
        "eval/reward_tempo_bin": sum(item["tempo_bin"] for item in all_rewards) / n,
        "eval/reward_tempo_bin_tolerant": sum(item["tempo_bin_tolerant"] for item in all_rewards) / n,
        "eval/reward_key_profile": sum(item["key_profile"] for item in all_rewards) / n,
        "eval/reward_meter_template": sum(item["meter_template"] for item in all_rewards) / n,
        "eval/reward_duration_balance": sum(item["duration_balance"] for item in all_rewards) / n,
        "eval/reward_note_density": sum(item["note_density"] for item in all_rewards) / n,
        "eval/reward_pitch_range": sum(item["pitch_range"] for item in all_rewards) / n,
        "eval/reward_pitch_diversity": sum(item["pitch_diversity"] for item in all_rewards) / n,
        "eval/reward_polyphony_balance": sum(item["polyphony_balance"] for item in all_rewards) / n,
        "eval/reward_duration_variety": sum(item["duration_variety"] for item in all_rewards) / n,
        "eval/reward_stepwise_motion": sum(item["stepwise_motion"] for item in all_rewards) / n,
        "eval/reward_repetition_balance": sum(item["repetition_balance"] for item in all_rewards) / n,
        "eval/valid_rate": sum(1 - item["invalid"] for item in all_rewards) / n,
        "eval/n_prompts": len(prompts),
        "eval/n_generations": n,
    }
    if all_critic_metrics:
        failure_rates = []
        for spec in critic_specs:
            summary.update(_critic_mean_summary(all_critic_metrics, spec.name))
            summary.update(_critic_group_summary(all_critic_metrics, spec.name, generations_per_prompt))
            failure_rates.append(
                sum(1.0 if item.get(f"{spec.name}_error") else 0.0 for item in all_critic_metrics) / max(len(all_critic_metrics), 1)
            )
        critic_success_rate = 1.0 - (sum(failure_rates) / len(failure_rates) if failure_rates else 0.0)
        summary["eval/observer_metric_success_rate"] = critic_success_rate
        summary["eval/critic_metric_success_rate"] = critic_success_rate
    summary.update(
        {
            f"eval/{key.replace('/', '_')}": value
            for key, value in summarize_text2midi_metrics(all_decoded_scores, all_captions).items()
        }
    )
    return summary
