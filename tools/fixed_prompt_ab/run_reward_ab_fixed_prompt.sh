#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-fixed_prompt_top1_len800_90x}"
REWARD_EXPERIMENT="${2:-ffn_key_moderate_density_drumratio_nomenter_fixedprompt_len800_debug}"
CRITICS="${3:-}"

GENERATIONS_JSON="outputs/fixed_prompt_ab/${RUN_NAME}/generations.json"
OUTPUT_JSON="outputs/fixed_prompt_ab/${RUN_NAME}/reward_scores_${REWARD_EXPERIMENT}.json"

if [[ ! -f "${GENERATIONS_JSON}" ]]; then
  echo "Generations JSON not found: ${GENERATIONS_JSON}" >&2
  exit 1
fi

CMD=(
  ./.venv/bin/python tools/fixed_prompt_ab/score_saved_midis_rewards.py
  --generations-json "${GENERATIONS_JSON}"
  --reward-experiment "${REWARD_EXPERIMENT}"
  --output-json "${OUTPUT_JSON}"
)

if [[ -n "${CRITICS}" ]]; then
  CMD+=(--critics "${CRITICS}")
fi

"${CMD[@]}"

echo "==> Internal reward A/B summary saved inside:"
echo "${OUTPUT_JSON}"

