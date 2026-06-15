#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "text2midi" / "prompts" / "grok_abc_melody_harmony_judge.txt"
DEFAULT_MIDI2ABC = PROJECT_ROOT / "tools" / "bin" / "midi2abc"
DEFAULT_API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4.3"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_FREE_GROK_MODEL = "x-ai/grok-4-fast:free"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
AITUNNEL_API_URL = "https://api.aitunnel.ru/v1/chat/completions"
AITUNNEL_DEFAULT_MODEL = "deepseek-v3.2"


@dataclass
class VoiceBlock:
    voice_id: str
    raw_lines: list[str]
    note_lines: list[str]
    is_percussion: bool = False

    def note_count(self) -> int:
        return len(re.findall(r"[A-Ga-g]", "\n".join(self.note_lines)))

    def should_keep(self, min_notes: int) -> bool:
        return not self.is_percussion and self.note_count() >= min_notes


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge MIDI files with Grok via ABC conversion.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi", help="Single MIDI file to judge.")
    source.add_argument("--input-root", help="Directory to scan for MIDI files.")
    parser.add_argument("--glob", default="**/generated.mid", help="Glob used with --input-root.")
    parser.add_argument("--output-dir", required=True, help="Where to save ABC, prompts, responses and aggregate results.")
    parser.add_argument("--template-file", default=str(DEFAULT_TEMPLATE), help="Prompt template with {{ABC}}.")
    parser.add_argument("--midi2abc-bin", default=str(DEFAULT_MIDI2ABC), help="Path to midi2abc executable.")
    parser.add_argument(
        "--provider",
        choices=["xai", "openrouter", "openrouter-free", "groq", "aitunnel"],
        default="xai",
        help="API provider preset. aitunnel uses OpenAI-compatible API at api.aitunnel.ru.",
    )
    parser.add_argument("--model", default=None, help="Grok model name. Defaults depend on --provider.")
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible chat completions URL. Defaults depend on --provider.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable containing the API key. Defaults depend on --provider.")
    parser.add_argument("--system-prompt", default="You are a strict but fair symbolic music judge.")
    parser.add_argument(
        "--transport",
        choices=["auto", "urllib", "openai"],
        default="auto",
        help="HTTP transport. 'openai' uses the official openai Python client against OpenAI-compatible endpoints.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of MIDI files, 0 means all.")
    parser.add_argument("--sleep", type=float, default=None, help="Delay between API calls.")
    parser.add_argument("--retries", type=int, default=4, help="Retries for rate limits/transient API errors.")
    parser.add_argument("--prepare-only", action="store_true", help="Only convert MIDI to ABC and write prompts.")
    parser.add_argument("--no-clean-abc", action="store_true", help="Do not remove tiny/percussion ABC voices.")
    parser.add_argument("--min-voice-notes", type=int, default=2)
    parser.add_argument("--max-voices", type=int, default=None, help="Keep the densest N voices after cleaning; 0 means no cap.")
    parser.add_argument("--max-abc-chars", type=int, default=None, help="Trim ABC if longer; 0 disables trimming.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run files that already have result.json.")
    args = parser.parse_args()
    apply_provider_defaults(args)
    return args


def apply_provider_defaults(args: argparse.Namespace) -> None:
    if args.provider == "openrouter-free":
        args.api_url = args.api_url or OPENROUTER_API_URL
        args.model = args.model or OPENROUTER_FREE_GROK_MODEL
        args.api_key_env = args.api_key_env or "OPENROUTER_API_KEY"
        args.max_tokens = args.max_tokens if args.max_tokens is not None else 140
        args.max_abc_chars = args.max_abc_chars if args.max_abc_chars is not None else 9000
        args.max_voices = args.max_voices if args.max_voices is not None else 10
        args.sleep = args.sleep if args.sleep is not None else 1.0
    elif args.provider == "openrouter":
        args.api_url = args.api_url or OPENROUTER_API_URL
        args.model = args.model or OPENROUTER_FREE_GROK_MODEL
        args.api_key_env = args.api_key_env or "OPENROUTER_API_KEY"
        args.max_tokens = args.max_tokens if args.max_tokens is not None else 180
        args.max_abc_chars = args.max_abc_chars if args.max_abc_chars is not None else 12000
        args.max_voices = args.max_voices if args.max_voices is not None else 12
        args.sleep = args.sleep if args.sleep is not None else 0.5
    elif args.provider == "groq":
        args.api_url = args.api_url or GROQ_API_URL
        args.model = args.model or GROQ_DEFAULT_MODEL
        args.api_key_env = args.api_key_env or "GROQ_API_KEY"
        args.max_tokens = args.max_tokens if args.max_tokens is not None else 120
        args.max_abc_chars = args.max_abc_chars if args.max_abc_chars is not None else 7000
        args.max_voices = args.max_voices if args.max_voices is not None else 8
        args.sleep = args.sleep if args.sleep is not None else 2.0
    elif args.provider == "aitunnel":
        args.api_url = args.api_url or AITUNNEL_API_URL
        args.model = args.model or AITUNNEL_DEFAULT_MODEL
        args.api_key_env = args.api_key_env or "AITUNNEL_API_KEY"
        args.transport = "openai" if args.transport == "auto" else args.transport
        args.max_tokens = args.max_tokens if args.max_tokens is not None else 180
        args.max_abc_chars = args.max_abc_chars if args.max_abc_chars is not None else 9000
        args.max_voices = args.max_voices if args.max_voices is not None else 10
        args.sleep = args.sleep if args.sleep is not None else 0.0
    else:
        args.api_url = args.api_url or DEFAULT_API_URL
        args.model = args.model or DEFAULT_MODEL
        args.api_key_env = args.api_key_env or "XAI_API_KEY"
        args.max_tokens = args.max_tokens if args.max_tokens is not None else 220
        args.max_abc_chars = args.max_abc_chars if args.max_abc_chars is not None else 14000
        args.max_voices = args.max_voices if args.max_voices is not None else 12
        args.sleep = args.sleep if args.sleep is not None else 0.0


def resolve_api_key(args: argparse.Namespace) -> str | None:
    candidates = [args.api_key_env]
    if args.provider.startswith("openrouter"):
        candidates.extend(["OPENROUTER_API_KEY", "GROK_API_KEY", "XAI_API_KEY"])
    elif args.provider == "groq":
        candidates.extend(["GROQ_API_KEY", "GROQ_API", "GROQ_KEY"])
    elif args.provider == "aitunnel":
        candidates.extend(["AITUNNEL_API_KEY", "AITUNNEL_KEY", "AITUNNEL_API", "API"])
    else:
        candidates.extend(["XAI_API_KEY", "GROK_API_KEY"])
    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        value = os.environ.get(name)
        if value:
            return value
    return None


def run_midi2abc(midi_path: Path, midi2abc_bin: Path) -> str:
    cmd = [str(midi2abc_bin), str(midi_path), "-gk"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"midi2abc failed for {midi_path}: {proc.stderr.strip() or proc.stdout.strip()}")
    text = proc.stdout.strip()
    if not text:
        raise RuntimeError(f"midi2abc produced empty ABC for {midi_path}")
    return text + "\n"


def clean_abc_text(text: str, min_voice_notes: int, max_voices: int) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    header_keep_prefixes = ("X:", "T:", "M:", "L:", "Q:", "K:")
    header: list[str] = []
    blocks: list[VoiceBlock] = []
    current: VoiceBlock | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("%") and not line.startswith("%%MIDI"):
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
            current = VoiceBlock(voice_id=line[2:].strip(), raw_lines=[line], note_lines=[])
            continue
        if current is None:
            if line.startswith(header_keep_prefixes):
                header.append(line)
            continue
        current.note_lines.append(line)

    if current is not None:
        blocks.append(current)

    kept = [block for block in blocks if block.should_keep(min_voice_notes)]
    kept.sort(key=lambda block: block.note_count(), reverse=True)
    if max_voices > 0:
        kept = kept[:max_voices]

    cleaned_lines = list(header)
    for block in kept:
        cleaned_lines.append(f"V:{block.voice_id}")
        cleaned_lines.extend(block.note_lines)

    stats = {
        "original_voice_count": len(blocks),
        "kept_voice_count": len(kept),
        "dropped_voice_ids": [block.voice_id for block in blocks if block not in kept],
        "kept_voice_ids": [block.voice_id for block in kept],
        "kept_voice_note_counts": {block.voice_id: block.note_count() for block in kept},
    }
    return "\n".join(cleaned_lines).strip() + "\n", stats


def trim_abc(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    keep_notice = "\n% ABC truncated for judge context length.\n"
    return text[: max(0, max_chars - len(keep_notice))].rstrip() + keep_notice, True


def build_prompt(template: str, abc_text: str) -> str:
    return template.replace("{{ABC}}", abc_text.strip())


def grok_chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider: str,
    retries: int,
    transport: str,
) -> dict[str, Any]:
    if transport in {"auto", "openai"}:
        try:
            return _chat_completion_openai_client(
                api_url=api_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                provider=provider,
                retries=retries,
            )
        except ModuleNotFoundError:
            if transport == "openai":
                raise RuntimeError(
                    "openai Python package is not installed. Install it or use --transport urllib."
                )
        except Exception:
            if transport == "openai":
                raise
    return _chat_completion_urllib(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        provider=provider,
        retries=retries,
    )


def _chat_completion_openai_client(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider: str,
    retries: int,
) -> dict[str, Any]:
    from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

    base_url = api_url.rsplit("/chat/completions", 1)[0]
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    extra_headers: dict[str, str] = {}
    if provider.startswith("openrouter"):
        extra_headers["HTTP-Referer"] = "https://github.com/humtech/Text2midi"
        extra_headers["X-Title"] = "Text2MIDI Grok Judge"

    for attempt in range(max(1, retries + 1)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers or None,
            )
            return response.model_dump()
        except RateLimitError as exc:
            if attempt >= retries:
                raise RuntimeError(f"{provider} API rate limit: {exc}") from exc
            delay = min(30.0, 2.0 * (attempt + 1))
            print(f"{provider} openai-client retry {attempt + 1}/{retries} after rate limit; sleeping {delay:.1f}s")
            time.sleep(delay)
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"{provider} API HTTP {status}: {exc}") from exc
            delay = min(30.0, 2.0 * (attempt + 1))
            print(f"{provider} openai-client retry {attempt + 1}/{retries} after HTTP {status}; sleeping {delay:.1f}s")
            time.sleep(delay)
        except APIConnectionError as exc:
            if attempt >= retries:
                raise RuntimeError(f"{provider} API connection error: {exc}") from exc
            delay = min(30.0, 2.0 * (attempt + 1))
            print(f"{provider} openai-client retry {attempt + 1}/{retries} after connection error; sleeping {delay:.1f}s")
            time.sleep(delay)

    raise RuntimeError(f"{provider} API failed after retries")


def _chat_completion_urllib(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider: str,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider.startswith("openrouter"):
        headers["HTTP-Referer"] = "https://github.com/humtech/Text2midi"
        headers["X-Title"] = "Text2MIDI Grok Judge"
    req = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
    for attempt in range(max(1, retries + 1)):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt >= retries:
                raise RuntimeError(f"{provider} API HTTP {exc.code}: {error_body}") from exc
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 2.0 * (attempt + 1)
            else:
                delay = min(30.0, 2.0 * (attempt + 1))
            print(f"{provider} API retry {attempt + 1}/{retries} after HTTP {exc.code}; sleeping {delay:.1f}s")
            time.sleep(delay)
        except urllib.error.URLError as exc:
            if attempt >= retries:
                raise RuntimeError(f"{provider} API connection error: {exc}") from exc
            delay = min(30.0, 2.0 * (attempt + 1))
            print(f"{provider} API retry {attempt + 1}/{retries} after connection error; sleeping {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"{provider} API failed after retries")


def extract_response_text(api_response: dict[str, Any]) -> str:
    choices = api_response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def parse_score_reason(text: str) -> tuple[float | None, str | None]:
    score_match = re.search(r"Score\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    reason_match = re.search(r"Reason\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    score = float(score_match.group(1)) if score_match else None
    if score is not None:
        score = max(1.0, min(10.0, score))
    reason = reason_match.group(1).strip() if reason_match else None
    return score, reason


def discover_midi_files(args: argparse.Namespace) -> list[Path]:
    if args.midi:
        return [resolve_path(args.midi)]
    root = resolve_path(args.input_root)
    paths = sorted(path for path in root.glob(args.glob) if path.is_file())
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    return paths


def sample_output_dir(base_dir: Path, midi_path: Path, input_root: Path | None, idx: int) -> Path:
    if input_root is not None:
        try:
            rel = midi_path.relative_to(input_root)
        except ValueError:
            rel = Path(midi_path.name)
        stem = "__".join(rel.with_suffix("").parts)
    else:
        stem = midi_path.stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return base_dir / f"{idx:04d}_{safe_stem}"


def judge_one(
    *,
    midi_path: Path,
    output_dir: Path,
    template: str,
    args: argparse.Namespace,
    api_key: str | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_abc = run_midi2abc(midi_path, resolve_path(args.midi2abc_bin))
    if args.no_clean_abc:
        abc = raw_abc
        cleaning_stats = {"cleaning_enabled": False}
    else:
        abc, cleaning_stats = clean_abc_text(raw_abc, args.min_voice_notes, args.max_voices)
        cleaning_stats["cleaning_enabled"] = True

    abc, was_truncated = trim_abc(abc, args.max_abc_chars)
    prompt = build_prompt(template, abc)

    (output_dir / "raw.abc").write_text(raw_abc, encoding="utf-8")
    (output_dir / "judge.abc").write_text(abc, encoding="utf-8")
    (output_dir / "judge_prompt.txt").write_text(prompt, encoding="utf-8")

    result: dict[str, Any] = {
        "midi_path": str(midi_path),
        "output_dir": str(output_dir),
        "model": args.model,
        "provider": args.provider,
        "api_url": args.api_url,
        "abc_path": str(output_dir / "judge.abc"),
        "prompt_path": str(output_dir / "judge_prompt.txt"),
        "abc_chars": len(abc),
        "prompt_chars": len(prompt),
        "abc_truncated": was_truncated,
        "cleaning_stats": cleaning_stats,
        "prepared_only": bool(args.prepare_only),
        "score": None,
        "reason": None,
        "response": None,
        "error": None,
    }

    if args.prepare_only:
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {args.api_key_env}=... "
            "or use --provider openrouter-free with OPENROUTER_API_KEY, or run with --prepare-only."
        )

    api_response = grok_chat_completion(
        api_url=args.api_url,
        api_key=api_key,
        model=args.model,
        system_prompt=args.system_prompt,
        user_prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        provider=args.provider,
        retries=args.retries,
        transport=args.transport,
    )
    response_text = extract_response_text(api_response)
    score, reason = parse_score_reason(response_text)
    result.update({
        "prepared_only": False,
        "response": response_text,
        "score": score,
        "reason": reason,
        "raw_api_response_path": str(output_dir / "raw_api_response.json"),
    })
    (output_dir / "raw_api_response.json").write_text(json.dumps(api_response, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "response.txt").write_text(response_text + "\n", encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(r["score"]) for r in results if r.get("score") is not None and r.get("error") is None]
    return {
        "count": len(results),
        "scored_count": len(scores),
        "error_count": sum(1 for r in results if r.get("error")),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }


def infer_group_and_pair_key(midi_path: Path, input_root: Path | None) -> tuple[str | None, str | None]:
    if input_root is None:
        return None, None
    try:
        rel = midi_path.relative_to(input_root)
    except ValueError:
        return None, None
    parts = rel.parts
    group = parts[0] if len(parts) >= 2 else None
    pair_key = parts[1] if len(parts) >= 3 else None
    return group, pair_key


def grouped_summaries(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(r["group"]) for r in results if r.get("group")})
    return {group: summarize([r for r in results if r.get("group") == group]) for group in groups}


def pairwise_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, dict[str, float]] = {}
    for result in results:
        group = result.get("group")
        pair_key = result.get("pair_key")
        score = result.get("score")
        if not group or not pair_key or score is None or result.get("error"):
            continue
        by_pair.setdefault(str(pair_key), {})[str(group)] = float(score)

    comparable = {key: value for key, value in by_pair.items() if "base" in value and "final" in value}
    wins = {"base": 0, "final": 0, "tie": 0}
    deltas: list[float] = []
    for scores in comparable.values():
        delta = scores["final"] - scores["base"]
        deltas.append(delta)
        if delta > 0:
            wins["final"] += 1
        elif delta < 0:
            wins["base"] += 1
        else:
            wins["tie"] += 1
    total = len(comparable)
    return {
        "pair_count": total,
        "wins": wins,
        "final_win_rate_excluding_ties": wins["final"] / max(1, wins["final"] + wins["base"]),
        "final_minus_base_mean": sum(deltas) / total if total else None,
        "final_minus_base_min": min(deltas) if deltas else None,
        "final_minus_base_max": max(deltas) if deltas else None,
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = resolve_path(args.input_root) if args.input_root else None
    midi_paths = discover_midi_files(args)
    template = resolve_path(args.template_file).read_text(encoding="utf-8")
    api_key = resolve_api_key(args)

    results: list[dict[str, Any]] = []
    jsonl_path = output_root / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for idx, midi_path in enumerate(midi_paths):
            out_dir = sample_output_dir(output_root / "samples", midi_path, input_root, idx)
            existing = out_dir / "result.json"
            if existing.exists() and not args.overwrite:
                result = json.loads(existing.read_text(encoding="utf-8"))
                print(f"[{idx + 1}/{len(midi_paths)}] reuse {midi_path} score={result.get('score')}")
            else:
                try:
                    result = judge_one(
                        midi_path=midi_path,
                        output_dir=out_dir,
                        template=template,
                        args=args,
                        api_key=api_key,
                    )
                    print(f"[{idx + 1}/{len(midi_paths)}] ok {midi_path} score={result.get('score')}")
                except Exception as exc:
                    result = {
                        "midi_path": str(midi_path),
                        "output_dir": str(out_dir),
                        "model": args.model,
                        "provider": args.provider,
                        "score": None,
                        "reason": None,
                        "response": None,
                        "error": str(exc),
                    }
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[{idx + 1}/{len(midi_paths)}] error {midi_path}: {exc}")

            group, pair_key = infer_group_and_pair_key(midi_path, input_root)
            result["group"] = group
            result["pair_key"] = pair_key
            results.append(result)
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            if args.sleep > 0 and idx + 1 < len(midi_paths) and not args.prepare_only:
                time.sleep(args.sleep)

    aggregate = {
        "model": args.model,
        "input_root": str(input_root) if input_root else None,
        "output_root": str(output_root),
        "prepare_only": bool(args.prepare_only),
        "summary": summarize(results),
        "group_summaries": grouped_summaries(results),
        "pairwise_summary": pairwise_summary(results),
        "results": results,
    }
    (output_root / "results.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
