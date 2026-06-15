from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

from core.config import train_config
from core.training.grpo import GRPOTrainer
from core.utils.runtime import load_env, resolve_device, seed_everything, setup_logging
from text2midi.adapter import Text2MidiAdapter
from text2midi.dataset import make_dataloader
from text2midi.prompting import parse_prompt_metadata

log = logging.getLogger(__name__)


def _apply_runtime_overrides(cfg, overrides: list[str]) -> None:
    if not overrides:
        return
    override_cfg = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(cfg, override_cfg)
    cfg.clear()
    cfg.merge_with(merged)


def _resolve_resume_checkpoint_dir(cfg, args) -> Path | None:
    if not getattr(args, "resume_from", None) and not getattr(args, "resume_latest", False):
        return None
    if getattr(args, "resume_from", None):
        path = Path(args.resume_from)
        if not path.is_absolute():
            path = Path(cfg.project_root) / path
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint path not found: {path}")
        return path.resolve()

    experiment_dir = Path(cfg.paths.checkpoint_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory exists yet for resume-latest: {experiment_dir}")
    step_dirs = sorted(
        [path for path in experiment_dir.iterdir() if path.is_dir() and path.name.startswith("step_")],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No saved step_* directories found in {experiment_dir}")
    return step_dirs[-1].resolve()


def _critic_group_diagnostics(
    reward_dicts: list[dict[str, float | str | None]],
    metric_name: str,
    group_size: int,
) -> dict[str, float]:
    top1_raw_values: list[float] = []
    top1_gap_values: list[float] = []
    for group_start in range(0, len(reward_dicts), group_size):
        group = reward_dicts[group_start : group_start + group_size]
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
        f"metric/{metric_name}_top1_raw": sum(top1_raw_values) / len(top1_raw_values) if top1_raw_values else 0.0,
        f"metric/{metric_name}_top1_gap": sum(top1_gap_values) / len(top1_gap_values) if top1_gap_values else 0.0,
    }


def _critic_metric_aggregates(
    reward_dicts: list[dict[str, float | str | None]],
    metric_name: str,
) -> dict[str, float]:
    metric_values = [float(item.get(metric_name, 0.0) or 0.0) for item in reward_dicts]
    raw_values = [float(item.get(f"{metric_name}_raw", 0.0) or 0.0) for item in reward_dicts]
    rank_values = [float(item.get(f"{metric_name}_rank_score", 0.0) or 0.0) for item in reward_dicts]
    pairwise_values = [float(item.get(f"{metric_name}_pairwise_win_rate", 0.0) or 0.0) for item in reward_dicts]
    failure_values = [1.0 if item.get(f"{metric_name}_error") else 0.0 for item in reward_dicts]
    n = max(len(reward_dicts), 1)
    return {
        f"metric/{metric_name}": sum(metric_values) / n,
        f"metric/{metric_name}_raw": sum(raw_values) / n,
        f"metric/{metric_name}_rank": sum(rank_values) / n,
        f"metric/{metric_name}_pairwise": sum(pairwise_values) / n,
        f"metrics/{metric_name}_success_rate": 1.0 - sum(failure_values) / n,
    }


def _save_step_prompts(cfg, step: int, captions: list[str]) -> None:
    prompt_dir = Path(cfg.paths.output_dir) / cfg.experiment / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, caption in enumerate(captions):
        metadata = parse_prompt_metadata(caption)
        prompt_path = prompt_dir / f"step_{step:05d}_prompt_{idx:02d}.txt"
        prompt_path.write_text(caption.strip() + "\n", encoding="utf-8")
        records.append(
            {
                "step": int(step),
                "prompt_index": int(idx),
                "prompt_path": str(prompt_path.resolve()),
                "prompt": caption,
                "metadata": metadata,
            }
        )

    step_json_path = prompt_dir / f"step_{step:05d}.json"
    step_json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (prompt_dir / "prompts.jsonl").open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_step_midis(
    cfg,
    step: int,
    captions: list[str],
    scores: list,
    reward_scores: list,
    track_selection_stats: list[dict],
    reward_dicts: list[dict[str, float | str | None]],
) -> None:
    group_size = int(cfg.grpo.num_rollouts)
    midi_root = Path(cfg.paths.output_dir) / cfg.experiment / "midis" / f"step_{step:05d}"
    midi_root.mkdir(parents=True, exist_ok=True)

    records = []
    for prompt_idx, caption in enumerate(captions):
        prompt_dir = midi_root / f"prompt_{prompt_idx:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(caption.strip() + "\n", encoding="utf-8")
        metadata = parse_prompt_metadata(caption)

        for rollout_idx in range(group_size):
            flat_idx = prompt_idx * group_size + rollout_idx
            score = scores[flat_idx] if flat_idx < len(scores) else None
            reward_score = reward_scores[flat_idx] if flat_idx < len(reward_scores) else None
            selection_stats = track_selection_stats[flat_idx] if flat_idx < len(track_selection_stats) else {}
            reward = reward_dicts[flat_idx] if flat_idx < len(reward_dicts) else {}
            raw_midi_path = prompt_dir / f"rollout_{rollout_idx:02d}_raw.mid"
            selected_midi_path = prompt_dir / f"rollout_{rollout_idx:02d}_selected.mid"
            decode_ok = score is not None
            selected_ok = reward_score is not None
            error = None
            if score is not None:
                try:
                    score.dump_midi(str(raw_midi_path))
                except Exception as exc:  # noqa: BLE001 - keep training alive while preserving the failure.
                    decode_ok = False
                    error = str(exc)
            if reward_score is not None:
                try:
                    reward_score.dump_midi(str(selected_midi_path))
                except Exception as exc:  # noqa: BLE001
                    selected_ok = False
                    error = str(exc)

            records.append(
                {
                    "step": int(step),
                    "prompt_index": int(prompt_idx),
                    "rollout_index": int(rollout_idx),
                    "prompt": caption,
                    "metadata": metadata,
                    "midi_path": str(selected_midi_path.resolve()) if selected_ok else None,
                    "selected_midi_path": str(selected_midi_path.resolve()) if selected_ok else None,
                    "raw_midi_path": str(raw_midi_path.resolve()) if decode_ok else None,
                    "decode_ok": bool(decode_ok),
                    "selected_ok": bool(selected_ok),
                    "error": error,
                    "track_selection": selection_stats,
                    "reward": dict(reward),
                }
            )

    step_manifest = midi_root / "manifest.json"
    step_manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_root = Path(cfg.paths.output_dir) / cfg.experiment
    with (manifest_root / "midis_manifest.jsonl").open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_training(args) -> None:
    cfg = train_config("text2midi", args.experiment)
    _apply_runtime_overrides(cfg, list(getattr(args, "overrides", []) or []))

    if args.num_rollouts:
        cfg.grpo.num_rollouts = args.num_rollouts
    if args.rollout_max_len:
        cfg.grpo.rollout_max_len = args.rollout_max_len
    if args.generation_chunk_size:
        cfg.grpo.generation_chunk_size = args.generation_chunk_size
    if args.update_mini_batch:
        cfg.grpo.update_mini_batch = args.update_mini_batch
    if args.lr:
        cfg.training.lr = args.lr
    if args.max_steps:
        cfg.training.max_steps = args.max_steps
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.save_every:
        cfg.training.save_every = args.save_every
    if args.synthetic:
        cfg.prompt.num_prompts_per_step = cfg.training.batch_size

    cfg.paths.checkpoint_dir = os.path.join(cfg.paths.checkpoint_dir, cfg.experiment)
    cfg.paths.log_dir = os.path.join(cfg.paths.log_dir, cfg.experiment)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    resume_checkpoint_dir = _resolve_resume_checkpoint_dir(cfg, args)

    setup_logging(os.path.join(cfg.paths.log_dir, "train.log"))
    load_env(cfg.project_root)
    seed_everything(cfg.training.seed)
    device = resolve_device()

    log.info("Experiment : %s", cfg.experiment)
    log.info("Description: %s", cfg.get("description", "-"))
    log.info("Objective  : %s", cfg.training.objective)
    log.info("Prompt mode: %s", "synthetic" if args.synthetic else "real captions")
    log.info("Device     : %s", device)
    log.info("Save every : %s steps", cfg.training.save_every)
    if resume_checkpoint_dir:
        log.info("Resume from: %s", resume_checkpoint_dir)

    OmegaConf.save(cfg, os.path.join(cfg.paths.checkpoint_dir, "config.yaml"))

    adapter = Text2MidiAdapter(
        cfg,
        device=device,
        with_lora=True,
        checkpoint_path=str(resume_checkpoint_dir) if resume_checkpoint_dir else None,
    )
    loader = make_dataloader(cfg, seed=cfg.training.seed)
    trainer = GRPOTrainer(cfg, adapter=adapter, device=device)
    step = trainer.load_state(resume_checkpoint_dir) if resume_checkpoint_dir else 0

    use_wandb = cfg.logging.use_wandb and not args.no_wandb
    if use_wandb:
        import wandb

        wandb_kwargs = {
            "project": os.environ.get("WANDB_PROJECT", cfg.logging.project),
            "name": os.environ.get(
                "WANDB_NAME",
                cfg.experiment if not resume_checkpoint_dir else f"{cfg.experiment}_resume_{step:05d}",
            ),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        wandb_run_id = os.environ.get("WANDB_RUN_ID")
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
        wandb.init(**wandb_kwargs)

    for _ in range(9999):
        for captions in loader:
            if step >= cfg.training.max_steps:
                break

            lr = trainer.set_lr_for_step(step)
            _save_step_prompts(cfg, step, captions)
            batch, reward_dicts = trainer.rollout(captions)
            if bool(cfg.training.get("save_rollout_midis", True)):
                _save_step_midis(
                    cfg,
                    step,
                    captions,
                    batch.scores,
                    batch.reward_scores,
                    batch.track_selection_stats,
                    reward_dicts,
                )
            metrics = trainer.update(batch)
            total = [item["total"] for item in reward_dicts]
            key = [item["key"] for item in reward_dicts]
            key_conditioned = [item["key_conditioned"] for item in reward_dicts]
            key_exact = [item["key_exact"] for item in reward_dicts]
            key_relative = [item["key_relative"] for item in reward_dicts]
            rhythm = [item["rhythm"] for item in reward_dicts]
            meter = [item["meter"] for item in reward_dicts]
            tempo = [item["tempo"] for item in reward_dicts]
            tempo_bin = [item["tempo_bin"] for item in reward_dicts]
            tempo_bin_tolerant = [item["tempo_bin_tolerant"] for item in reward_dicts]
            key_profile = [item["key_profile"] for item in reward_dicts]
            meter_template = [item["meter_template"] for item in reward_dicts]
            duration_balance = [item["duration_balance"] for item in reward_dicts]
            note_density = [item["note_density"] for item in reward_dicts]
            moderate_note_density = [item["moderate_note_density"] for item in reward_dicts]
            pitch_range = [item["pitch_range"] for item in reward_dicts]
            pitch_diversity = [item["pitch_diversity"] for item in reward_dicts]
            polyphony_balance = [item["polyphony_balance"] for item in reward_dicts]
            duration_variety = [item["duration_variety"] for item in reward_dicts]
            stepwise_motion = [item["stepwise_motion"] for item in reward_dicts]
            repetition_balance = [item["repetition_balance"] for item in reward_dicts]
            track_count = [item["track_count"] for item in reward_dicts]
            track_count_exact = [item["track_count_exact"] for item in reward_dicts]
            no_drums = [item["no_drums"] for item in reward_dicts]
            drum_note_ratio = [item["drum_note_ratio"] for item in reward_dicts]
            raw_track_count = [item.get("raw_track_count", 0.0) for item in reward_dicts]
            selected_track_count = [item.get("selected_track_count", 0.0) for item in reward_dicts]
            invalid = [item["invalid"] for item in reward_dicts]
            observer = [item.get("observer", 0.0) for item in reward_dicts]
            observer_raw = [item.get("observer_raw", 0.0) for item in reward_dicts]
            observer_failures = [1.0 if item.get("observer_error") else 0.0 for item in reward_dicts]
            metrics.update({"reward/total": sum(total) / len(total)})

            reward_metric_values = {
                "key": key,
                "key_conditioned": key_conditioned,
                "key_exact": key_exact,
                "key_relative": key_relative,
                "rhythm": rhythm,
                "meter": meter,
                "tempo": tempo,
                "tempo_bin": tempo_bin,
                "tempo_bin_tolerant": tempo_bin_tolerant,
                "key_profile": key_profile,
                "meter_template": meter_template,
                "duration_balance": duration_balance,
                "note_density": note_density,
                "moderate_note_density": moderate_note_density,
                "pitch_range": pitch_range,
                "pitch_diversity": pitch_diversity,
                "polyphony_balance": polyphony_balance,
                "duration_variety": duration_variety,
                "stepwise_motion": stepwise_motion,
                "repetition_balance": repetition_balance,
                "track_count": track_count,
                "track_count_exact": track_count_exact,
                "no_drums": no_drums,
                "drum_note_ratio": drum_note_ratio,
                "observer": observer,
            }
            reward_weight_keys = {
                "key": "key_weight",
                "key_conditioned": "key_conditioned_weight",
                "key_exact": "key_exact_weight",
                "key_relative": "key_relative_weight",
                "rhythm": "rhythm_weight",
                "meter": "meter_weight",
                "tempo": "tempo_weight",
                "tempo_bin": "tempo_bin_weight",
                "tempo_bin_tolerant": "tempo_bin_tolerant_weight",
                "key_profile": "key_profile_weight",
                "meter_template": "meter_template_weight",
                "duration_balance": "duration_balance_weight",
                "note_density": "note_density_weight",
                "moderate_note_density": "moderate_note_density_weight",
                "pitch_range": "pitch_range_weight",
                "pitch_diversity": "pitch_diversity_weight",
                "polyphony_balance": "polyphony_balance_weight",
                "duration_variety": "duration_variety_weight",
                "stepwise_motion": "stepwise_motion_weight",
                "repetition_balance": "repetition_balance_weight",
                "track_count": "track_count_weight",
                "track_count_exact": "track_count_exact_weight",
                "no_drums": "no_drums_weight",
                "drum_note_ratio": "drum_note_ratio_weight",
                "observer": "observer_weight",
            }
            for metric_name, values in reward_metric_values.items():
                weight_key = reward_weight_keys[metric_name]
                if float(cfg.reward.get(weight_key, 0.0)) != 0.0:
                    metrics[f"reward/{metric_name}"] = sum(values) / len(values)

            track_selection_cfg = cfg.reward.get("track_selection", {}) or {}
            if bool(track_selection_cfg.get("enabled", False)):
                metrics["metric/raw_track_count"] = sum(raw_track_count) / len(raw_track_count)
                metrics["metric/selected_track_count"] = sum(selected_track_count) / len(selected_track_count)

            if bool(cfg.reward.get("observer_weight", 0.0)) != 0.0:
                metrics["reward/observer_raw"] = sum(observer_raw) / len(observer_raw)
                metrics["metrics/observer_success_rate"] = 1.0 - sum(observer_failures) / len(observer_failures)

            if bool(cfg.reward.get("invalid_enabled", True)):
                metrics["penalty/invalid"] = sum(invalid) / len(invalid)
                metrics["metrics/valid_rate"] = 1.0 - sum(invalid) / len(invalid)

            metrics["train/lr"] = lr
            critic_failure_rates = []
            for spec in trainer.metric_critics:
                metrics.update(_critic_metric_aggregates(reward_dicts, spec.name))
                metrics.update(_critic_group_diagnostics(reward_dicts, spec.name, cfg.grpo.num_rollouts))
                critic_failure_rates.append(
                    sum(1.0 if item.get(f"{spec.name}_error") else 0.0 for item in reward_dicts) / max(len(reward_dicts), 1)
                )
            if critic_failure_rates:
                metrics["metrics/critic_metric_success_rate"] = 1.0 - sum(critic_failure_rates) / len(critic_failure_rates)
            else:
                metrics["metrics/critic_metric_success_rate"] = 1.0

            if step % 10 == 0:
                valid_rate = float(metrics.get("metrics/valid_rate", 1.0))
                log.info(
                    "step=%4d loss=%.4f reward=%.3f kl=%.5f grad=%.3f ratio=%.3f lr=%.2e skipped=%.0f valid=%.1f%%",
                    step,
                    metrics["loss"],
                    metrics["reward/total"],
                    metrics["kl"],
                    metrics.get("grad_norm", 0.0),
                    metrics.get("mean_ratio", 1.0),
                    lr,
                    metrics.get("skipped", 0.0),
                    100.0 * valid_rate,
                )

            if use_wandb:
                import wandb

                wandb.log(metrics, step=step)

            step += 1

            if step % cfg.training.save_every == 0:
                trainer.save(step)

        if step >= cfg.training.max_steps:
            break

    trainer.save(step)
    if use_wandb:
        import wandb

        wandb.finish()
    log.info("Done.")
