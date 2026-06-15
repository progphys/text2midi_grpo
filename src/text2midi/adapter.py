from __future__ import annotations

import logging
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import T5Tokenizer

from core.interfaces.model_adapter import BaseModelAdapter

log = logging.getLogger(__name__)


class Text2MidiAdapter(BaseModelAdapter):
    def __init__(self, cfg, device: torch.device, with_lora: bool = False, checkpoint_path: str | None = None):
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.adaptation_mode = self._resolve_adaptation_mode(cfg, checkpoint_path, with_lora)
        self._trainable_param_names: set[str] = set()

        repo_path = Path(cfg.paths.model_repo)
        sys.path.insert(0, str(repo_path))

        from model.transformer_model import Transformer

        repo_cfg = OmegaConf.load(repo_path / "configs/config.yaml")
        model_cfg = repo_cfg.model.text2midi_model

        with open(Path(cfg.paths.weights_dir) / "vocab_remi.pkl", "rb") as handle:
            self.r_tokenizer = pickle.load(handle)
        vocab_size = len(self.r_tokenizer)

        self.t5_tokenizer = T5Tokenizer.from_pretrained(cfg.paths.tokenizer_dir)

        def build_base():
            model = Transformer(
                n_vocab=vocab_size,
                d_model=model_cfg.decoder_d_model,
                nhead=model_cfg.decoder_num_heads,
                max_len=model_cfg.decoder_max_sequence_length,
                num_decoder_layers=model_cfg.decoder_num_layers,
                dim_feedforward=model_cfg.decoder_intermediate_size,
                use_moe=model_cfg.use_moe,
                device=device,
            )
            model.load_state_dict(
                torch.load(
                    Path(cfg.paths.weights_dir) / "pytorch_model.bin",
                    map_location=device,
                    weights_only=False,
                )
            )
            return model

        base = build_base()
        if self.adaptation_mode == "lora":
            if self._qlora_enabled(cfg):
                base = self._quantize_for_qlora(base, cfg)
            if checkpoint_path:
                base = PeftModel.from_pretrained(base, checkpoint_path, is_trainable=True)
            else:
                lora_cfg = self._build_lora_config(cfg)
                base = get_peft_model(base, lora_cfg)
            self.model = base.to(device)
        elif self.adaptation_mode == "ffn":
            self._freeze_all(base)
            self._enable_direct_finetune(base, cfg)
            if checkpoint_path:
                self._load_trainable_state(base, checkpoint_path)
            self.model = base.to(device)
        else:
            self.model = base.to(device)
            self._trainable_param_names = {name for name, param in self.model.named_parameters() if param.requires_grad}

        if self.adaptation_mode != "base" and not any(param.requires_grad for param in self.model.parameters()):
            raise RuntimeError(
                f"No trainable parameters found after initializing adaptation_mode={self.adaptation_mode}"
                + (f" from checkpoint={checkpoint_path}" if checkpoint_path else "")
            )

        self.model.eval()

        ref_model = build_base()
        if self.adaptation_mode == "lora" and self._qlora_enabled(cfg) and bool(cfg.qlora.get("quantize_reference", True)):
            ref_model = self._quantize_for_qlora(ref_model, cfg)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _qlora_enabled(cfg) -> bool:
        return "qlora" in cfg and bool(cfg.qlora.get("enabled", False))

    @staticmethod
    def _resolve_adaptation_mode(cfg, checkpoint_path: str | None, with_lora: bool) -> str:
        if not with_lora:
            return "base"
        checkpoint_cfg_path = None
        if checkpoint_path:
            checkpoint_cfg_path = Path(checkpoint_path) / "config.yaml"
            if checkpoint_cfg_path.exists():
                try:
                    checkpoint_cfg = OmegaConf.load(checkpoint_cfg_path)
                    mode = str(checkpoint_cfg.get("adaptation", {}).get("mode", "") or "").strip().lower()
                    if mode:
                        return mode
                except Exception:
                    log.exception("Failed to infer adaptation mode from %s", checkpoint_cfg_path)
        mode = str(cfg.get("adaptation", {}).get("mode", "lora") or "lora").strip().lower()
        return mode

    @staticmethod
    def _build_lora_config(cfg) -> LoraConfig:
        kwargs = dict(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            target_modules=list(cfg.lora.target_modules),
        )
        layers = cfg.lora.get("layers_to_transform")
        if layers is not None:
            kwargs["layers_to_transform"] = list(layers)
            kwargs["layers_pattern"] = "layers"
        return LoraConfig(**kwargs)

    @staticmethod
    def _freeze_all(model: torch.nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad_(False)

    def _enable_direct_finetune(self, model: torch.nn.Module, cfg) -> None:
        target_names = set(cfg.get("adaptation", {}).get("trainable_modules", ["linear1", "linear2"]))
        target_layers_cfg = cfg.get("adaptation", {}).get("trainable_layer_indices")
        target_layers = None if target_layers_cfg is None else {int(idx) for idx in target_layers_cfg}
        trainable: set[str] = set()
        for module_name, module in model.named_modules():
            leaf_name = module_name.rsplit(".", 1)[-1]
            if leaf_name not in target_names:
                continue
            if target_layers is not None:
                layer_index = self._extract_decoder_layer_index(module_name)
                if layer_index is None or layer_index not in target_layers:
                    continue
            for param_name, param in module.named_parameters(recurse=False):
                param.requires_grad_(True)
                trainable.add(f"{module_name}.{param_name}" if module_name else param_name)
        if not trainable:
            raise RuntimeError(
                f"Direct FFN finetune was requested, but no parameters matched adaptation.trainable_modules={sorted(target_names)}"
            )
        self._trainable_param_names = trainable
        log.info("Direct finetune enabled for %d parameter tensors", len(sorted(trainable)))

    @staticmethod
    def _extract_decoder_layer_index(module_name: str) -> int | None:
        parts = module_name.split(".")
        for idx in range(len(parts) - 2):
            if parts[idx] == "decoder" and parts[idx + 1] == "layers":
                try:
                    return int(parts[idx + 2])
                except ValueError:
                    return None
        return None

    def _load_trainable_state(self, model: torch.nn.Module, checkpoint_path: str) -> None:
        state_path = Path(checkpoint_path) / "trainable_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Direct finetune checkpoint not found: {state_path}")
        payload = torch.load(state_path, map_location=self.device, weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        current_state = model.state_dict()
        unexpected = [name for name in state_dict.keys() if name not in current_state]
        missing = [name for name in self._trainable_param_names if name not in state_dict]
        current_state.update({name: tensor for name, tensor in state_dict.items() if name in current_state})
        model.load_state_dict(current_state, strict=True)
        if missing:
            log.warning("Missing trainable keys while loading direct finetune state: %s", missing[:8])
        if unexpected:
            log.warning("Unexpected keys while loading direct finetune state: %s", unexpected[:8])

    def _quantize_for_qlora(self, model: torch.nn.Module, cfg) -> torch.nn.Module:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "QLoRA requires bitsandbytes. Install project requirements or run "
                "`pip install bitsandbytes==0.45.5`."
            ) from exc

        compute_dtype_name = str(cfg.qlora.get("compute_dtype", "bfloat16"))
        compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)
        quant_type = str(cfg.qlora.get("quant_type", "nf4"))
        compress_statistics = bool(cfg.qlora.get("compress_statistics", True))
        target_names = set(cfg.qlora.get("quantize_modules", []))

        def should_quantize(module_name: str, module: nn.Module) -> bool:
            if not isinstance(module, nn.Linear):
                return False
            if not target_names:
                return True
            return module_name.rsplit(".", 1)[-1] in target_names

        replacements: list[tuple[nn.Module, str, nn.Linear]] = []
        for module_name, module in model.named_modules():
            for child_name, child in module.named_children():
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                if should_quantize(full_name, child):
                    replacements.append((module, child_name, child))

        for parent, child_name, child in replacements:
            quantized = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )
            quantized.weight = bnb.nn.Params4bit(
                child.weight.detach().cpu(),
                requires_grad=False,
                compress_statistics=compress_statistics,
                quant_type=quant_type,
            )
            if child.bias is not None:
                quantized.bias = nn.Parameter(child.bias.detach().cpu(), requires_grad=False)
            setattr(parent, child_name, quantized)

        if not replacements:
            raise RuntimeError("QLoRA was enabled, but no Linear modules matched qlora.quantize_modules.")
        return model

    def tokenize_captions(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.t5_tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def generate_repeated(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_return_sequences: int,
        max_len: int,
        temperature: float,
    ) -> torch.Tensor:
        repeated_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        repeated_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
        chunk_size = int(
            self.cfg.get("grpo", {}).get(
                "generation_chunk_size",
                self.cfg.get("inference", {}).get("generation_chunk_size", 4),
            )
        )
        chunk_size = max(1, chunk_size)
        total_sequences = repeated_ids.shape[0]
        total_chunks = (total_sequences + chunk_size - 1) // chunk_size

        log.info(
            "Generation started: total_sequences=%d chunk_size=%d total_chunks=%d max_len=%d temperature=%.3f",
            total_sequences,
            chunk_size,
            total_chunks,
            max_len,
            temperature,
        )

        self.model.eval()
        chunks = []
        for start in range(0, repeated_ids.shape[0], chunk_size):
            stop = start + chunk_size
            chunk_index = start // chunk_size + 1
            log.info(
                "Generating chunk %d/%d: sequences=%d",
                chunk_index,
                total_chunks,
                repeated_ids[start:stop].shape[0],
            )
            try:
                chunk = self._generate_batch(
                    model=self.model,
                    input_ids=repeated_ids[start:stop],
                    attention_mask=repeated_mask[start:stop],
                    max_len=max_len,
                    temperature=temperature,
                )
            except Exception:
                log.exception(
                    "Chunk generation failed, falling back to per-sequence generation for chunk %d/%d",
                    chunk_index,
                    total_chunks,
                )
                chunk = self._generate_batch_fallback(
                    input_ids=repeated_ids[start:stop],
                    attention_mask=repeated_mask[start:stop],
                    max_len=max_len,
                    temperature=temperature,
                )
            chunks.append(chunk)

        log.info("Generation finished: total_sequences=%d", total_sequences)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _generate_batch(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_len: int,
        temperature: float,
    ) -> torch.Tensor:
        generated = torch.full(
            (input_ids.shape[0], 1),
            1,
            dtype=torch.long,
            device=self.device,
        )
        temperature = max(float(temperature), 1e-6)
        for _ in range(max_len):
            output = model(
                input_ids,
                attention_mask,
                generated,
                memory_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=True,
                memory_is_causal=False,
            )
            if isinstance(output, tuple):
                output = output[0]
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sums > 0, probs / row_sums.clamp(min=1e-8), torch.ones_like(probs) / probs.shape[-1])
            next_tokens = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_tokens), dim=1)
        return generated[:, 1:]

    def _generate_batch_fallback(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_len: int,
        temperature: float,
    ) -> torch.Tensor:
        pad = torch.ones(max_len, dtype=torch.long, device=self.device)
        sequences = []
        for idx in range(input_ids.shape[0]):
            try:
                output = self.model.generate(
                    input_ids[idx : idx + 1],
                    attention_mask[idx : idx + 1],
                    max_len=max_len,
                    temperature=temperature,
                )
                sequence = output[0]
                if torch.isnan(sequence.float()).any() or torch.isinf(sequence.float()).any():
                    sequence = pad
            except Exception:
                sequence = pad
            sequences.append(sequence)

        return torch.stack(sequences, dim=0)

    def sequence_token_log_probs(self, model, input_ids, attention_mask, generated):
        tgt_in = generated[:, :-1]
        tgt_out = generated[:, 1:]

        logits = model(input_ids, attention_mask, tgt_in)
        if isinstance(logits, tuple):
            logits = logits[0]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(dim=-1, index=tgt_out.unsqueeze(-1)).squeeze(-1)
        token_lp = torch.nan_to_num(token_lp, nan=0.0, posinf=0.0, neginf=0.0)

        mask = (tgt_out != 0).float()
        return token_lp, mask

    def _sequence_log_probs(self, model, input_ids, attention_mask, generated):
        token_lp, mask = self.sequence_token_log_probs(model, input_ids, attention_mask, generated)
        lengths = mask.sum(dim=-1).clamp(min=1)
        return (token_lp * mask).sum(dim=-1) / lengths

    def score_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated: torch.Tensor,
        use_reference: bool = False,
    ) -> torch.Tensor:
        model = self.ref_model if use_reference else self.model
        return self._sequence_log_probs(model, input_ids, attention_mask, generated)

    def score_sequence_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated: torch.Tensor,
        use_reference: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.ref_model if use_reference else self.model
        return self.sequence_token_log_probs(model, input_ids, attention_mask, generated)

    def decode_scores(self, generated: torch.Tensor) -> list:
        scores = []
        for sequence in generated:
            try:
                scores.append(self.r_tokenizer.decode(sequence.tolist()))
            except Exception:
                scores.append(None)
        return scores

    def trainable_parameters(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    def save_adapter(self, output_dir: str) -> None:
        if self.adaptation_mode == "lora":
            self.model.save_pretrained(output_dir)
            return
        state_dict = {
            name: tensor.detach().cpu()
            for name, tensor in self.model.state_dict().items()
            if name in self._trainable_param_names
        }
        torch.save(
            {
                "mode": self.adaptation_mode,
                "trainable_param_names": sorted(self._trainable_param_names),
                "state_dict": state_dict,
            },
            Path(output_dir) / "trainable_state.pt",
        )

    def generate_to_file(
        self,
        caption: str,
        max_len: int,
        temperature: float,
        output_dir: str,
        to_wav: bool = False,
        soundfont_path: str | None = None,
    ) -> tuple[str, str | None]:
        input_ids, attention_mask = self.tokenize_captions([caption])
        generated = self.generate_repeated(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=1,
            max_len=max_len,
            temperature=temperature,
        )
        score = self.decode_scores(generated)[0]
        if score is None:
            raise RuntimeError("Model output could not be decoded into a MIDI score.")

        os.makedirs(output_dir, exist_ok=True)
        safe_name = re.sub(r"[^\w\s-]", "", caption[:60]).strip().replace(" ", "_")
        midi_path = os.path.join(output_dir, f"{safe_name}.mid")
        score.dump_midi(midi_path)

        wav_path = None
        if to_wav:
            wav_path = midi_path.replace(".mid", ".wav")
            soundfont = soundfont_path or "/usr/share/soundfonts/FluidR3_GM.sf2"
            result = subprocess.run(
                ["fluidsynth", "-ni", soundfont, midi_path, "-F", wav_path, "-r", "44100"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                wav_path = None

        return midi_path, wav_path
