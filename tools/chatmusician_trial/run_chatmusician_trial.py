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
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "text2midi" / "prompts" / "chatmusician_trial_melody_harmony_judge.txt"


@dataclass
class VoiceBlock:
    voice_id: str
    raw_lines: list[str]
    note_lines: list[str]
    is_percussion: bool = False

    def note_count(self) -> int:
        return len(re.findall(r"[A-Ga-g]", "\n".join(self.note_lines)))

    def rest_count(self) -> int:
        return len(re.findall(r"\bz\b", "\n".join(self.note_lines)))

    def should_keep(self) -> bool:
        if self.is_percussion:
            return False
        note_count = self.note_count()
        if note_count < 2:
            return False
        if note_count == 0:
            return False
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Clean ABC and run a single ChatMusician trial.")
    parser.add_argument("--abc-input", required=True, help="Path to source ABC/text file.")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned files and optional model output.")
    parser.add_argument("--template-file", default=str(DEFAULT_TEMPLATE), help="Prompt template with {{ABC}} placeholder.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo or local model path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do-sample", action="store_true")
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
    voice_blocks: list[VoiceBlock] = []
    current: VoiceBlock | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("T:"):
            continue
        if line.startswith("%"):
            continue
        if line.startswith("%%MIDI"):
            if current and "channel 10" in line:
                current.is_percussion = True
            continue
        if line.startswith("%%clef"):
            continue
        if line.startswith("V:"):
            if current is not None:
                voice_blocks.append(current)
            current = VoiceBlock(voice_id=line[2:].strip(), raw_lines=[line], note_lines=[])
            continue
        if line.startswith(header_keep_prefixes) and current is None:
            header.append(line)
            continue
        if current is None:
            continue
        current.note_lines.append(line)

    if current is not None:
        voice_blocks.append(current)

    kept_blocks = [block for block in voice_blocks if block.should_keep()]
    cleaned_lines = list(header)
    stats = {
        "original_voice_count": len(voice_blocks),
        "kept_voice_count": len(kept_blocks),
        "dropped_voice_ids": [block.voice_id for block in voice_blocks if not block.should_keep()],
        "kept_voice_ids": [block.voice_id for block in kept_blocks],
    }

    for block in kept_blocks:
        cleaned_lines.append(f"V:{block.voice_id}")
        cleaned_lines.extend(block.note_lines)

    return "\n".join(cleaned_lines).strip() + "\n", stats


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(template: str, abc_text: str) -> str:
    return template.replace("{{ABC}}", abc_text.strip())


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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
    model_candidate = resolve_path(args.model)
    if model_candidate.exists():
        model_path = str(model_candidate)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_text = abc_input.read_text(encoding="utf-8")
    cleaned_abc, stats = clean_abc_text(source_text)
    template = load_template(template_file)
    prompt = build_prompt(template, cleaned_abc)

    cleaned_path = output_dir / "cleaned_abc.txt"
    prompt_path = output_dir / "trial_prompt.txt"
    meta_path = output_dir / "cleaning_stats.json"
    cleaned_path.write_text(cleaned_abc, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.prepare_only:
        print(json.dumps({
            "cleaned_abc_path": str(cleaned_path),
            "trial_prompt_path": str(prompt_path),
            "cleaning_stats_path": str(meta_path),
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
            temperature=args.temperature,
            do_sample=bool(args.do_sample),
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    result = {
        "model": model_path,
        "device": device,
        "cleaned_abc_path": str(cleaned_path),
        "trial_prompt_path": str(prompt_path),
        "cleaning_stats": stats,
        "response": response,
    }
    result_path = output_dir / "chatmusician_trial_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
