#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config import inference_config
from text2midi.adapter import Text2MidiAdapter
from text2midi.prompting import generate_batch, parse_prompt_metadata

from tools.ab_test_final_observer.common import (
    default_output_dir,
    dump_json,
    get_device,
    latest_checkpoint_dir,
    load_experiment_cfg,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paired A/B samples for base and finetuned models.")
    parser.add_argument("--experiment", default="final_observer_only_ffn_finetune_18rollouts")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional explicit finetuned checkpoint dir.")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--samples-per-model", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-preset", default="broad")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--generation-chunk-size", type=int, default=None)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("ab_%Y%m%d_%H%M%S")
    output_root = resolve_path(args.output_dir) or default_output_dir(run_name)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = resolve_path(args.checkpoint_dir) or latest_checkpoint_dir(args.experiment)
    cfg = load_experiment_cfg(args.experiment)
    infer_cfg = inference_config("text2midi")
    cfg.inference = infer_cfg.inference
    if args.generation_chunk_size is not None:
        cfg.inference.generation_chunk_size = int(args.generation_chunk_size)
    if "grpo" in cfg:
        cfg.grpo.generation_chunk_size = int(
            args.generation_chunk_size
            if args.generation_chunk_size is not None
            else cfg.inference.generation_chunk_size
        )
    max_len = int(args.max_len or cfg.inference.max_len)
    temperature = float(args.temperature or cfg.inference.temperature)
    device = get_device()

    prompts = generate_batch(args.num_prompts, seed=args.seed, preset=args.prompt_preset)
    (output_root / "prompts.json").write_text(
        __import__("json").dumps(prompts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Output root: {output_root.resolve()}", flush=True)
    print(f"Prompt preset: {args.prompt_preset}", flush=True)
    print(f"Finetuned checkpoint: {checkpoint_dir.resolve()}", flush=True)
    print(
        f"Generating {args.num_prompts} prompts x {args.samples_per_model} samples x 2 models",
        flush=True,
    )
    print(f"Prompt batch size: {args.prompt_batch_size}", flush=True)

    base_adapter = Text2MidiAdapter(cfg, device=device, with_lora=False)
    final_adapter = Text2MidiAdapter(cfg, device=device, with_lora=True, checkpoint_path=str(checkpoint_dir))

    manifest: list[dict] = []
    prompt_batch_size = max(1, int(args.prompt_batch_size))
    total_prompts = len(prompts)
    for batch_start in range(0, total_prompts, prompt_batch_size):
        batch_stop = min(total_prompts, batch_start + prompt_batch_size)
        prompt_batch = prompts[batch_start:batch_stop]
        print(
            f"[{batch_start + 1:03d}-{batch_stop:03d}/{total_prompts:03d}] generating batch of {len(prompt_batch)} prompts",
            flush=True,
        )

        prompt_dirs: list[tuple[int, str, Path, Path]] = []
        for local_idx, prompt in enumerate(prompt_batch):
            prompt_index = batch_start + local_idx
            prompt_dir_base = output_root / "base" / f"prompt_{prompt_index:03d}"
            prompt_dir_final = output_root / "final" / f"prompt_{prompt_index:03d}"
            prompt_dir_base.mkdir(parents=True, exist_ok=True)
            prompt_dir_final.mkdir(parents=True, exist_ok=True)
            (prompt_dir_base / "prompt.txt").write_text(prompt, encoding="utf-8")
            (prompt_dir_final / "prompt.txt").write_text(prompt, encoding="utf-8")
            prompt_dirs.append((prompt_index, prompt, prompt_dir_base, prompt_dir_final))

        input_ids, attention_mask = base_adapter.tokenize_captions(prompt_batch)
        base_generated = base_adapter.generate_repeated(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=args.samples_per_model,
            max_len=max_len,
            temperature=temperature,
        )
        final_generated = final_adapter.generate_repeated(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=args.samples_per_model,
            max_len=max_len,
            temperature=temperature,
        )
        base_scores = base_adapter.decode_scores(base_generated)
        final_scores = final_adapter.decode_scores(final_generated)

        for model_name, decoded_scores, prompt_dir_idx in (
            ("base", base_scores, 2),
            ("final", final_scores, 3),
        ):
            for local_idx, (prompt_index, prompt, prompt_dir_base, prompt_dir_final) in enumerate(prompt_dirs):
                prompt_dir = prompt_dir_base if model_name == "base" else prompt_dir_final
                prompt_metadata = parse_prompt_metadata(prompt)
                for sample_index in range(args.samples_per_model):
                    flat_index = local_idx * args.samples_per_model + sample_index
                    score = decoded_scores[flat_index]
                    if score is None:
                        midi_path = None
                        decode_ok = False
                    else:
                        midi_path = prompt_dir / f"sample_{sample_index:02d}.mid"
                        score.dump_midi(str(midi_path))
                        midi_path = str(midi_path.resolve())
                        decode_ok = True
                    manifest.append(
                        {
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "prompt_metadata": prompt_metadata,
                            "model_name": model_name,
                            "sample_index": sample_index,
                            "midi_path": midi_path,
                            "decode_ok": decode_ok,
                        }
                    )

    run_config = {
        "experiment": args.experiment,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "num_prompts": args.num_prompts,
        "samples_per_model": args.samples_per_model,
        "seed": args.seed,
        "prompt_preset": args.prompt_preset,
        "max_len": max_len,
        "temperature": temperature,
        "generation_chunk_size": int(cfg.inference.generation_chunk_size),
        "prompt_batch_size": prompt_batch_size,
        "device": str(device),
        "output_root": str(output_root.resolve()),
    }
    dump_json(output_root / "run_config.json", run_config)
    dump_json(output_root / "generations.json", manifest)
    print(str((output_root / "generations.json").resolve()))


if __name__ == "__main__":
    main()
