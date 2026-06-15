#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "m-a-p/ChatMusician"
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "text2midi" / "prompts" / "chatmusician_trial_closed_judge.txt"


@dataclass
class VoiceBlock:
    voice_id: str
    note_lines: list[str]
    is_percussion: bool = False

    def note_count(self) -> int:
        return len(re.findall(r"[A-Ga-g]", "\n".join(self.note_lines)))

    def should_keep(self) -> bool:
        return (not self.is_percussion) and self.note_count() >= 2


def parse_args():
    parser = argparse.ArgumentParser(description="Run a closed-format ChatMusician trial on cleaned ABC.")
    parser.add_argument("--abc-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--template-file", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def clean_abc_text(text: str) -> tuple[str, dict]:
    lines = text.splitlines()
    header_keep_prefixes = ("X:", "M:", "L:", "Q:", "K:")
    header: list[str] = []
    blocks: list[VoiceBlock] = []
    current: VoiceBlock | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("T:") or line.startswith("%"):
            continue
        if line.startswith("%%MIDI"):
            if current and "channel 10" in line:
                current.is_percussion = True
            continue
        if line.startswith("%%clef"):
            continue
        if line.startswith("V:"):
            if current is not None:
                blocks.append(current)
            current = VoiceBlock(voice_id=line[2:].strip(), note_lines=[])
            continue
        if current is None:
            if line.startswith(header_keep_prefixes):
                header.append(line)
            continue
        current.note_lines.append(line)

    if current is not None:
        blocks.append(current)

    kept = [b for b in blocks if b.should_keep()]
    cleaned_lines = list(header)
    stats = {
        "original_voice_count": len(blocks),
        "kept_voice_count": len(kept),
        "dropped_voice_ids": [b.voice_id for b in blocks if not b.should_keep()],
        "kept_voice_ids": [b.voice_id for b in kept],
    }
    for b in kept:
        cleaned_lines.append(f"V:{b.voice_id}")
        cleaned_lines.extend(b.note_lines)
    return "\n".join(cleaned_lines).strip() + "\n", stats


def build_prompt(template: str, abc_text: str) -> str:
    return template.replace("{{ABC}}", abc_text.strip())


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_dtype(name: str):
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, name)


def main():
    args = parse_args()
    abc_input = resolve_path(args.abc_input)
    output_dir = resolve_path(args.output_dir)
    template_file = resolve_path(args.template_file)
    model_path = args.model
    local_model = resolve_path(args.model)
    if local_model.exists():
        model_path = str(local_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_text = abc_input.read_text(encoding="utf-8")
    cleaned_abc, stats = clean_abc_text(source_text)
    template = template_file.read_text(encoding="utf-8")
    prompt = build_prompt(template, cleaned_abc)

    (output_dir / "cleaned_abc.txt").write_text(cleaned_abc, encoding="utf-8")
    (output_dir / "trial_prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "cleaning_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.prepare_only:
        print(json.dumps({
            "cleaned_abc_path": str((output_dir / 'cleaned_abc.txt').resolve()),
            "trial_prompt_path": str((output_dir / 'trial_prompt.txt').resolve()),
            "cleaning_stats_path": str((output_dir / 'cleaning_stats.json').resolve()),
            "stats": stats,
        }, ensure_ascii=False, indent=2))
        return

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    payload = {
        "model": model_path,
        "device": device,
        "cleaned_abc_path": str((output_dir / "cleaned_abc.txt").resolve()),
        "trial_prompt_path": str((output_dir / "trial_prompt.txt").resolve()),
        "cleaning_stats": stats,
        "response": response,
    }
    (output_dir / "chatmusician_closed_trial_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
