#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-fixed_prompt_top1_len800_90x}"
RESULTS_JSON="outputs/aitunnel_judge/${RUN_NAME}/results.json"
OUTPUT_JSON="outputs/aitunnel_judge/${RUN_NAME}/multi_model_ab_summary.json"

if [[ ! -f "${RESULTS_JSON}" ]]; then
  echo "Results JSON not found: ${RESULTS_JSON}" >&2
  echo "Run LLM judge first:" >&2
  echo "  ./tools/fixed_prompt_ab/run_llm_judge_fixed_prompt.sh ${RUN_NAME}" >&2
  exit 1
fi

./.venv/bin/python tools/fixed_prompt_ab/multi_model_judge_summary.py \
  --results-json "${RESULTS_JSON}" \
  --output-json "${OUTPUT_JSON}"

echo "==> Multi-model A/B summary:"
echo "${OUTPUT_JSON}"

