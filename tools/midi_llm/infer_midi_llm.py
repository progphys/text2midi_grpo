#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "slseanwu/MIDI-LLM_Llama-3.2-1B"


def parse_args():
    parser = argparse.ArgumentParser(description="Run MIDI-LLM inference on symbolic music text such as ABC.")
    parser.add_argument("--input-file", default=None, help="Path to text input file (.abc/.txt).")
    parser.add_argument("--input-text", default=None, help="Raw symbolic music text.")
    parser.add_argument("--question", required=True, help="Judge question or instruction.")
    parser.add_argument("--model-path", default=None, help="Local model path or HF repo id.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--context-prompt-file", default=None, help="Optional original generation prompt.")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _resolve_dtype(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return getattr(torch, name)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _read_text_arg(project_root: Path, input_file: str | None, input_text: str | None) -> str:
    if input_text:
        return input_text
    if input_file:
        path = Path(input_file)
        if not path.is_absolute():
            path = project_root / path
        return path.read_text(encoding="utf-8")
    raise ValueError("Specify --input-file or --input-text.")


def _build_prompt(symbolic_text: str, question: str, generation_prompt: str | None) -> str:
    parts = [
        "You are an expert symbolic music evaluator.",
    ]
    if generation_prompt:
        parts.extend([
            "",
            "Original generation prompt:",
            generation_prompt.strip(),
        ])
    parts.extend([
        "",
        "Symbolic music input:",
        symbolic_text.strip(),
        "",
        question.strip(),
    ])
    return "\n".join(parts).strip() + "\n"


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    symbolic_text = _read_text_arg(project_root, args.input_file, args.input_text)
    generation_prompt = None
    if args.context_prompt_file:
        prompt_path = Path(args.context_prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = project_root / prompt_path
        generation_prompt = prompt_path.read_text(encoding="utf-8")

    model_path = args.model_path
    if model_path is None:
        local_model = project_root / "models" / "midi_llm" / "checkpoints" / "model"
        model_path = str(local_model) if local_model.exists() else DEFAULT_MODEL
    else:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            local = project_root / model_path
            if local.exists():
                model_path = str(local)

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    prompt = _build_prompt(symbolic_text, args.question, generation_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": bool(args.do_sample),
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        output = model.generate(**inputs, **generation_kwargs)

    answer_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    payload = {
        "model_path": model_path,
        "device": device,
        "question": args.question,
        "response": answer,
        "prompt_chars": len(prompt),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
