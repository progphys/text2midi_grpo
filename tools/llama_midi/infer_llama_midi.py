#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import symusic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "dx2102/llama-midi"


def parse_args():
    parser = argparse.ArgumentParser(description="Run llama-midi inference on model-native symbolic music text.")
    parser.add_argument("--midi-file", default=None, help="Path to input MIDI file to preprocess with symusic.")
    parser.add_argument("--input-file", default=None, help="Path to preprocessed text input.")
    parser.add_argument("--input-text", default=None, help="Raw model-native symbolic text.")
    parser.add_argument("--question", required=True, help="Question or instruction for the model.")
    parser.add_argument("--model-path", default=None, help="Local model path or HF repo id.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--dump-preprocessed-text", default=None, help="Optional path to save MIDI-derived text.")
    return parser.parse_args()


def _resolve_dtype(name: str):
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, name)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path


def _preprocess_midi(path: Path) -> str:
    score = symusic.Score(str(path), ttype="Second")
    score = score.copy()

    for track in score.tracks:
        notes = track.notes
        pedals = track.pedals
        track.pedals = []
        j = 0
        for note in notes:
            while j < len(pedals) and pedals[j].time + pedals[j].duration < note.time:
                j += 1
            if j < len(pedals) and pedals[j].time <= note.time <= pedals[j].time + pedals[j].duration:
                note.duration = max(
                    note.duration,
                    pedals[j].time + pedals[j].duration - note.time,
                )

    notes = []
    for track in score.tracks:
        instrument = "drum" if track.is_drum else str(track.program)
        for note in track.notes:
            notes.append((note.time, note.duration, note.pitch, note.velocity, instrument))

    # Deduplicate near-identical notes the same way the model card suggests.
    notes = list({
        (time, duration, pitch): (time, duration, pitch, velocity, instrument)
        for time, duration, pitch, velocity, instrument in notes
    }.values())
    notes.sort(key=lambda x: (x[0], -x[2]))

    lines = ["pitch duration wait velocity instrument"]
    previous_start_ms = 0
    for time, duration, pitch, velocity, instrument in notes:
        start_ms = int(round(float(time) * 1000))
        duration_ms = max(1, int(round(float(duration) * 1000)))
        wait_ms = max(0, start_ms - previous_start_ms)
        model_velocity = max(1, min(31, int(round(int(velocity) / 4))))
        lines.append(f"{int(pitch)} {duration_ms} {wait_ms} {model_velocity} {instrument}")
        previous_start_ms = start_ms

    return "\n".join(lines)


def _read_text_arg(project_root: Path, midi_file: str | None, input_file: str | None, input_text: str | None) -> str:
    if input_text:
        return input_text
    if midi_file:
        return _preprocess_midi(_resolve_path(project_root, midi_file))
    if input_file:
        path = _resolve_path(project_root, input_file)
        return path.read_text(encoding="utf-8")
    raise ValueError("Specify --midi-file, --input-file, or --input-text.")


def _build_prompt(symbolic_text: str, question: str) -> str:
    parts = [
        "You are an expert symbolic music understanding assistant.",
        "",
        "MIDI-derived symbolic text:",
        symbolic_text.strip(),
        "",
        question.strip(),
    ]
    return "\n".join(parts).strip() + "\n"


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    symbolic_text = _read_text_arg(project_root, args.midi_file, args.input_file, args.input_text)

    if args.dump_preprocessed_text:
        dump_path = _resolve_path(project_root, args.dump_preprocessed_text)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(symbolic_text, encoding="utf-8")

    model_path = args.model_path
    if model_path is None:
        local_model = project_root / "models" / "llama_midi" / "checkpoints" / "model"
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

    prompt = _build_prompt(symbolic_text, args.question)
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
