#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-fixed_prompt_top1_len800_90x}"
INPUT_ROOT="outputs/fixed_prompt_ab/${RUN_NAME}"
OUTPUT_DIR="outputs/aitunnel_judge/${RUN_NAME}"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Input root not found: ${INPUT_ROOT}" >&2
  exit 1
fi

MIDI_COUNT="$(find "${INPUT_ROOT}" -name '*.mid' | wc -l)"
echo "==> MIDI files found: ${MIDI_COUNT}"
echo "==> Input : ${INPUT_ROOT}"
echo "==> Output: ${OUTPUT_DIR}"

./.venv/bin/python tools/grok_judge/grok_midi_judge.py \
  --provider aitunnel \
  --input-root "${INPUT_ROOT}" \
  --glob "**/*.mid" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite

echo "==> LLM judge results:"
echo "${OUTPUT_DIR}/results.json"

