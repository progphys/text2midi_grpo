from __future__ import annotations

import json
import re
from pathlib import Path

from core.config import inference_config
from core.utils.runtime import resolve_device, setup_logging
from text2midi.adapter import Text2MidiAdapter
from text2midi.prompting import generate_batch, parse_prompt_metadata


def run_base_probe(args) -> None:
    cfg = inference_config("text2midi")
    setup_logging()
    device = resolve_device()

    num_prompts = int(args.num_prompts or 8)
    seed = int(args.seed or 42)
    preset = str(args.preset or "melody_accompaniment_narrow")
    max_len = int(args.max_len or cfg.inference.max_len)
    temperature = float(args.temperature or cfg.inference.temperature)

    output_root = Path(args.output_dir) if args.output_dir else Path(cfg.paths.output_dir) / f"base_probe_{preset}"
    output_root.mkdir(parents=True, exist_ok=True)

    prompts = generate_batch(num_prompts, seed=seed, preset=preset)
    adapter = Text2MidiAdapter(cfg, device=device, with_lora=False)

    manifest: list[dict[str, object]] = []
    for idx, prompt in enumerate(prompts):
        midi_path, _wav_path = adapter.generate_to_file(
            caption=prompt,
            max_len=max_len,
            temperature=temperature,
            output_dir=str(output_root),
            to_wav=False,
        )
        safe_name = re.sub(r"[^\w\s-]", "", prompt[:60]).strip().replace(" ", "_")
        target_path = output_root / f"{idx:02d}_{safe_name}.mid"
        Path(midi_path).replace(target_path)
        manifest.append(
            {
                "index": idx,
                "prompt": prompt,
                "metadata": parse_prompt_metadata(prompt),
                "midi_path": str(target_path.resolve()),
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "preset": preset,
                "num_prompts": num_prompts,
                "seed": seed,
                "max_len": max_len,
                "temperature": temperature,
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Preset: {preset}")
    print(f"Output dir: {output_root.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
