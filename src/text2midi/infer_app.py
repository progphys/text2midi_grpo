from __future__ import annotations

from core.config import inference_config
from core.utils.runtime import resolve_device, setup_logging
from text2midi.adapter import Text2MidiAdapter


def run_inference(args) -> None:
    cfg = inference_config("text2midi")
    setup_logging()
    device = resolve_device()

    caption = args.caption or cfg.inference.caption
    max_len = args.max_len or cfg.inference.max_len
    temperature = args.temperature or cfg.inference.temperature
    to_wav = args.to_wav or cfg.inference.to_wav

    adapter = Text2MidiAdapter(cfg, device=device, with_lora=False)
    midi_path, wav_path = adapter.generate_to_file(
        caption=caption,
        max_len=max_len,
        temperature=temperature,
        output_dir=cfg.paths.output_dir,
        to_wav=to_wav,
        soundfont_path=cfg.inference.soundfont_path,
    )

    print(f"Device: {device}")
    print(f"Caption: {caption}")
    print(f"Saved MIDI: {midi_path}")
    if to_wav and wav_path:
        print(f"Saved WAV: {wav_path}")
    elif to_wav:
        print("fluidsynth or soundfont unavailable, skipping WAV conversion.")
