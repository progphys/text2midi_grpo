from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.critics.observer.projection import project_to_observer_format


class ObserverCriticError(RuntimeError):
    """Raised when the external observer critic cannot be invoked."""


@dataclass
class ObserverItem:
    id: str
    midi_path: str
    key: str
    mode: str
    bpm: float
    meter_numerator: int
    meter_denominator: int


def build_payload(items: list[ObserverItem]) -> dict[str, Any]:
    return {"items": [asdict(item) for item in items]}


def normalize_observer_score(raw_score: float, center: float = -20.0, scale: float = 5.0) -> float:
    """Map observer score to [0, 1] so it can be mixed with simpler rewards later."""
    if scale <= 0:
        raise ValueError("scale must be > 0")
    z = (raw_score - center) / scale
    return 1.0 / (1.0 + math.exp(-z))


class ObserverCritic:
    def __init__(
        self,
        project_root: str | Path,
        checkpoint_path: str | Path,
        python_executable: str | None = None,
        script_path: str | Path | None = None,
        chord_weights_yaml: str | Path | None = None,
        batch_size: int = 32,
        device: str = "auto",
        continue_on_error: bool = True,
        pretty: bool = True,
        project_midi: bool = False,
        projected_output_dir: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.checkpoint_path = self._resolve_path(checkpoint_path)
        self.script_path = self._resolve_path(script_path) if script_path else self.project_root / "src" / "core" / "critics" / "observer" / "infer.py"
        self.python_executable = self._resolve_python_executable(python_executable)
        self.chord_weights_yaml = self._resolve_path(chord_weights_yaml) if chord_weights_yaml else None
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.continue_on_error = continue_on_error
        self.pretty = pretty
        self.project_midi = project_midi
        self.projected_output_dir = self._resolve_path(projected_output_dir) if projected_output_dir else self.project_root / "outputs" / "observer_projected"

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _resolve_device(self, value: str) -> str:
        requested = str(value or "auto").strip().lower()
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _resolve_python_executable(self, value: str | None) -> str:
        candidates: list[str] = []
        if value:
            candidates.append(str(self._resolve_path(value)))
        candidates.append(sys.executable)

        env_python = os.environ.get("TEXT2MIDI_PYTHON")
        if env_python:
            candidates.append(env_python)
        for name in ("python3.10", "python3", "python"):
            resolved = shutil.which(name)
            if resolved:
                candidates.append(resolved)

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return sys.executable

    def score_items(self, items: list[ObserverItem]) -> dict[str, Any]:
        if not self.script_path.exists():
            raise ObserverCriticError(f"Observer script not found: {self.script_path}")
        if not self.checkpoint_path.exists():
            raise ObserverCriticError(f"Observer checkpoint not found: {self.checkpoint_path}")

        payload_items = items
        if self.project_midi:
            payload_items = self._project_items(items)
        payload = build_payload(payload_items)
        with tempfile.TemporaryDirectory(prefix="observer_critic_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_json = tmp_dir_path / "observer_input.json"
            output_json = tmp_dir_path / "observer_output.json"
            input_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            env = dict(**os.environ)
            pythonpath_parts = [str(self.project_root), str(self.project_root / "src")]
            if env.get("PYTHONPATH"):
                pythonpath_parts.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(pythonpath_parts)

            cmd = self._build_command(input_json=input_json, output_json=output_json, device=self.device)
            result = self._run_subprocess(cmd, env=env)
            if result.returncode != 0 and self.device != "cpu" and "out of memory" in result.stderr.lower():
                fallback_cmd = self._build_command(input_json=input_json, output_json=output_json, device="cpu")
                result = self._run_subprocess(fallback_cmd, env=env)
                cmd = fallback_cmd
            if result.returncode != 0:
                raise ObserverCriticError(
                    "Observer critic failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

            return json.loads(output_json.read_text(encoding="utf-8"))

    def _build_command(self, input_json: Path, output_json: Path, device: str) -> list[str]:
        cmd = [
            self.python_executable,
            str(self.script_path),
            "--input-json",
            str(input_json),
            "--checkpoint",
            str(self.checkpoint_path),
            "--output-json",
            str(output_json),
            "--device",
            device,
            "--batch-size",
            str(self.batch_size),
        ]
        if self.chord_weights_yaml:
            cmd.extend(["--chord-weights-yaml", str(self.chord_weights_yaml)])
        if self.continue_on_error:
            cmd.append("--continue-on-error")
        if self.pretty:
            cmd.append("--pretty")
        return cmd

    def _run_subprocess(self, cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
        )

    def _project_items(self, items: list[ObserverItem]) -> list[ObserverItem]:
        projected_items: list[ObserverItem] = []
        self.projected_output_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            src_path = Path(item.midi_path)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.id)).strip("_") or src_path.stem
            projected_path = self.projected_output_dir / f"{safe_id}_{src_path.stem}_observer.mid"
            try:
                project_to_observer_format(src_path, projected_path)
            except Exception:
                if not self.continue_on_error:
                    raise
                projected_items.append(item)
                continue
            projected_items.append(
                ObserverItem(
                    id=item.id,
                    midi_path=str(projected_path),
                    key=item.key,
                    mode=item.mode,
                    bpm=item.bpm,
                    meter_numerator=item.meter_numerator,
                    meter_denominator=item.meter_denominator,
                )
            )
        return projected_items

    def filter_results(
        self,
        payload: dict[str, Any],
        min_score: float | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        results = list(payload.get("results", []))
        results = [row for row in results if row.get("score") is not None]
        results.sort(key=lambda row: (-float(row["score"]), int(row["index"])))
        if min_score is not None:
            results = [row for row in results if float(row["score"]) >= min_score]
        if top_k is not None:
            results = results[:top_k]
        return results

    @classmethod
    def from_config(cls, project_root: str | Path, cfg) -> "ObserverCritic":
        return cls(
            project_root=project_root,
            checkpoint_path=cfg.checkpoint_path,
            python_executable=cfg.get("python_executable"),
            script_path=cfg.get("script_path"),
            chord_weights_yaml=cfg.get("chord_weights_yaml"),
            batch_size=int(cfg.get("batch_size", 32)),
            device=str(cfg.get("device", "auto")),
            continue_on_error=bool(cfg.get("continue_on_error", True)),
            pretty=bool(cfg.get("pretty", True)),
            project_midi=bool(cfg.get("project_midi", False)),
            projected_output_dir=cfg.get("projected_output_dir"),
        )
