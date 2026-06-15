from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from core.config import inference_config, train_config
from core.critics.observer_client import ObserverCritic, ObserverItem
from core.evaluation.text2midi_metrics import summarize_text2midi_metrics
from core.rewards import batch_rewards
from core.utils.runtime import resolve_device, setup_logging
from text2midi.adapter import Text2MidiAdapter
from text2midi.prompting import generate_batch, parse_prompt_metadata


def _resolve_checkpoint_dir(project_root: Path, checkpoint_path: str | None, experiment: str | None) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        return path

    if not experiment:
        raise ValueError("Provide either --checkpoint-path or --experiment for the fine-tuned model.")

    experiment_dir = project_root / "outputs" / "checkpoints" / experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment checkpoint directory not found: {experiment_dir}")

    step_dirs = sorted(
        [path for path in experiment_dir.iterdir() if path.is_dir() and path.name.startswith("step_")],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not step_dirs:
        raise FileNotFoundError(
            f"No saved step directories found in {experiment_dir}. "
            "Expected something like step_00200."
        )
    return step_dirs[-1]


def _build_critics(project_root: Path) -> list[tuple[str, ObserverCritic]]:
    critic_specs = [
        ("observer_default", project_root / "configs" / "text2midi" / "reward" / "observer.yaml"),
        ("observer_fixed_pairs", project_root / "configs" / "text2midi" / "reward" / "observer_fixed_pairs.yaml"),
        ("final_observer", project_root / "configs" / "text2midi" / "reward" / "final_observer.yaml"),
        ("final_final", project_root / "configs" / "text2midi" / "reward" / "final_final.yaml"),
    ]
    critics: list[tuple[str, ObserverCritic]] = []
    for critic_name, config_path in critic_specs:
        cfg = OmegaConf.load(config_path)
        critics.append((critic_name, ObserverCritic.from_config(project_root, cfg.reward.observer_critic)))
    return critics


def _score_with_critics(
    critics: list[tuple[str, ObserverCritic]],
    prompts: list[str],
    midi_paths: list[str],
    side: str,
) -> tuple[list[dict[str, float | str | None]], dict[str, dict]]:
    items: list[ObserverItem] = []
    metadata_available: list[bool] = []
    for idx, (prompt, midi_path) in enumerate(zip(prompts, midi_paths)):
        metadata = parse_prompt_metadata(prompt)
        if metadata is None:
            metadata_available.append(False)
            continue
        metadata_available.append(True)
        items.append(
            ObserverItem(
                id=f"{side}_{idx:03d}",
                midi_path=midi_path,
                key=str(metadata["key"]),
                mode=str(metadata["mode"]),
                bpm=float(metadata["bpm"]),
                meter_numerator=int(metadata["meter_numerator"]),
                meter_denominator=int(metadata["meter_denominator"]),
            )
        )

    per_item_scores: list[dict[str, float | str | None]] = [{} for _ in prompts]
    raw_payloads: dict[str, dict] = {}

    for critic_name, critic in critics:
        payload = critic.score_items(items)
        raw_payloads[critic_name] = payload
        for list_idx, row in enumerate(payload.get("results", [])):
            item_id = items[list_idx].id
            prompt_idx = int(item_id.rsplit("_", 1)[-1])
            per_item_scores[prompt_idx][critic_name] = row.get("score")
            if row.get("error") is not None:
                per_item_scores[prompt_idx][f"{critic_name}_error"] = row.get("error")

    for idx, ok in enumerate(metadata_available):
        if not ok:
            for critic_name, _critic in critics:
                per_item_scores[idx][critic_name] = None
                per_item_scores[idx][f"{critic_name}_error"] = "prompt_metadata_unavailable"

    return per_item_scores, raw_payloads


def _generate_to_file_with_score(
    adapter: Text2MidiAdapter,
    caption: str,
    max_len: int,
    temperature: float,
    output_dir: Path,
) -> tuple[str, object]:
    input_ids, attention_mask = adapter.tokenize_captions([caption])
    generated = adapter.generate_repeated(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=1,
        max_len=max_len,
        temperature=temperature,
    )
    score = adapter.decode_scores(generated)[0]
    if score is None:
        raise RuntimeError("Model output could not be decoded into a MIDI score.")

    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\s-]", "", caption[:60]).strip().replace(" ", "_")
    midi_path = output_dir / f"{safe_name}.mid"
    score.dump_midi(str(midi_path))
    return str(midi_path.resolve()), score


def _summarize_rewards(scores: list, prompts: list[str], reward_cfg) -> dict[str, float]:
    rows = batch_rewards(scores, reward_cfg, captions=prompts)
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {
        key: sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows)
        for key in keys
    }


def _with_compare_critic_rank_reward(
    base_rewards: dict[str, float],
    tuned_rewards: dict[str, float],
    base_scores: list[dict[str, float | str | None]] | None,
    tuned_scores: list[dict[str, float | str | None]] | None,
    reward_cfg,
) -> dict[str, dict[str, float]]:
    observer_weight = float(reward_cfg.get("observer_weight", 0.0))
    critic_cfg = reward_cfg.get("observer_critic")
    if not critic_cfg or not bool(critic_cfg.get("enabled", False)) or observer_weight == 0.0:
        return {"base": base_rewards, "finetuned": tuned_rewards}

    checkpoint_path = str(critic_cfg.get("checkpoint_path", ""))
    critic_name = "observer_fixed_pairs" if "observer_fixed_pairs" in checkpoint_path else "observer_default"
    base_scores = base_scores or []
    tuned_scores = tuned_scores or []
    reward_signal = str(critic_cfg.get("reward_signal", "rank_score"))
    base_rank_scores: list[float] = []
    tuned_rank_scores: list[float] = []
    base_pairwise_scores: list[float] = []
    tuned_pairwise_scores: list[float] = []

    for base_row, tuned_row in zip(base_scores, tuned_scores):
        base_raw = base_row.get(critic_name)
        tuned_raw = tuned_row.get(critic_name)
        if base_raw is None or tuned_raw is None:
            base_rank_scores.append(0.0)
            tuned_rank_scores.append(0.0)
            base_pairwise_scores.append(0.0)
            tuned_pairwise_scores.append(0.0)
            continue
        if float(tuned_raw) > float(base_raw):
            base_rank_scores.append(0.0)
            tuned_rank_scores.append(1.0)
            base_pairwise_scores.append(0.0)
            tuned_pairwise_scores.append(1.0)
        elif float(tuned_raw) < float(base_raw):
            base_rank_scores.append(1.0)
            tuned_rank_scores.append(0.0)
            base_pairwise_scores.append(1.0)
            tuned_pairwise_scores.append(0.0)
        else:
            base_rank_scores.append(0.5)
            tuned_rank_scores.append(0.5)
            base_pairwise_scores.append(0.5)
            tuned_pairwise_scores.append(0.5)

    def enrich(rewards: dict[str, float], rank_scores: list[float], pairwise_scores: list[float]) -> dict[str, float]:
        out = dict(rewards)
        rank_mean = sum(rank_scores) / len(rank_scores) if rank_scores else 0.0
        pairwise_mean = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
        symbolic_total = float(out.get("total", 0.0) or 0.0)
        out["critic_compare_rank"] = rank_mean
        out["critic_compare_pairwise"] = pairwise_mean
        out["total_symbolic"] = symbolic_total
        out["total_with_critic_compare_rank"] = symbolic_total + observer_weight * rank_mean
        out["total_with_critic_compare_pairwise"] = symbolic_total + observer_weight * pairwise_mean
        out["critic_compare_signal"] = reward_signal
        return out

    return {
        "base": enrich(base_rewards, base_rank_scores, base_pairwise_scores),
        "finetuned": enrich(tuned_rewards, tuned_rank_scores, tuned_pairwise_scores),
    }


def run_compare_inference(args) -> None:
    cfg = inference_config("text2midi")
    setup_logging()
    device = resolve_device()
    project_root = Path(cfg.project_root)

    num_prompts = args.num_prompts or 10
    seed = args.seed if args.seed is not None else 42
    max_len = args.max_len or cfg.inference.max_len
    temperature = args.temperature or cfg.inference.temperature

    checkpoint_dir = _resolve_checkpoint_dir(
        project_root=project_root,
        checkpoint_path=args.checkpoint_path,
        experiment=args.experiment,
    )
    reward_cfg = train_config("text2midi", args.experiment).reward if args.experiment else cfg.reward

    run_name = args.run_name or datetime.now().strftime("compare_%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "comparisons" / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    prompts = generate_batch(num_prompts, seed=seed)

    base_dir = output_root / "base"
    tuned_dir = output_root / "finetuned"
    base_dir.mkdir(parents=True, exist_ok=True)
    tuned_dir.mkdir(parents=True, exist_ok=True)

    base_adapter = Text2MidiAdapter(cfg, device=device, with_lora=False)
    tuned_adapter = Text2MidiAdapter(
        cfg,
        device=device,
        with_lora=True,
        checkpoint_path=str(checkpoint_dir),
    )

    manifest: list[dict[str, str | int]] = []
    base_paths: list[str] = []
    tuned_paths: list[str] = []
    base_symbolic_scores = []
    tuned_symbolic_scores = []
    for idx, prompt in enumerate(prompts):
        base_midi_path, base_score = _generate_to_file_with_score(
            adapter=base_adapter,
            caption=prompt,
            max_len=max_len,
            temperature=temperature,
            output_dir=base_dir,
        )
        tuned_midi_path, tuned_score = _generate_to_file_with_score(
            adapter=tuned_adapter,
            caption=prompt,
            max_len=max_len,
            temperature=temperature,
            output_dir=tuned_dir,
        )
        base_paths.append(base_midi_path)
        tuned_paths.append(tuned_midi_path)
        base_symbolic_scores.append(base_score)
        tuned_symbolic_scores.append(tuned_score)

        manifest.append(
            {
                "index": idx,
                "prompt": prompt,
                "metadata": parse_prompt_metadata(prompt),
                "base_midi": base_paths[-1],
                "finetuned_midi": tuned_paths[-1],
            }
        )

    critic_payloads: dict[str, dict] = {}
    base_scores = None
    tuned_scores = None
    if not getattr(args, "skip_critics", False):
        critics = _build_critics(project_root)
        base_scores, base_payloads = _score_with_critics(critics, prompts, base_paths, side="base")
        tuned_scores, tuned_payloads = _score_with_critics(critics, prompts, tuned_paths, side="finetuned")
        critic_payloads["base"] = base_payloads
        critic_payloads["finetuned"] = tuned_payloads
        for idx, item in enumerate(manifest):
            item["base_scores"] = base_scores[idx]
            item["finetuned_scores"] = tuned_scores[idx]

    base_reward_summary = _summarize_rewards(base_symbolic_scores, prompts, reward_cfg)
    tuned_reward_summary = _summarize_rewards(tuned_symbolic_scores, prompts, reward_cfg)
    reward_summaries = _with_compare_critic_rank_reward(
        base_rewards=base_reward_summary,
        tuned_rewards=tuned_reward_summary,
        base_scores=base_scores,
        tuned_scores=tuned_scores,
        reward_cfg=reward_cfg,
    )

    summary = {
        "device": str(device),
        "num_prompts": num_prompts,
        "seed": seed,
        "temperature": temperature,
        "max_len": max_len,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "base_output_dir": str(base_dir.resolve()),
        "finetuned_output_dir": str(tuned_dir.resolve()),
        "critics_enabled": not getattr(args, "skip_critics", False),
        "text2midi_metrics": {
            "base": summarize_text2midi_metrics(base_symbolic_scores, prompts),
            "finetuned": summarize_text2midi_metrics(tuned_symbolic_scores, prompts),
        },
        "reward_metrics": reward_summaries,
        "items": manifest,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if critic_payloads:
        (output_root / "critic_payloads.json").write_text(
            json.dumps(critic_payloads, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Device: {device}")
    print(f"Fine-tuned checkpoint: {checkpoint_dir.resolve()}")
    print(f"Output root: {output_root.resolve()}")
    print(f"Generated prompt pairs: {len(manifest)}")
    for side, rewards in reward_summaries.items():
        total_key = "total_with_critic_compare_rank" if "total_with_critic_compare_rank" in rewards else "total"
        print(f"{side} {total_key}: {float(rewards.get(total_key, 0.0)):.4f}")
    print(f"Manifest: {(output_root / 'manifest.json').resolve()}")
    if critic_payloads:
        print(f"Critic payloads: {(output_root / 'critic_payloads.json').resolve()}")
