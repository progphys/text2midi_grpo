#!/usr/bin/env bash
set -euo pipefail

./.venv/bin/python tools/fixed_prompt_ab/generate_one_prompt_samples.py \
  --selection-json outputs/model_registry/fixed_prompt_best_checkpoints.json \
  --run-name fixed_prompt_top1_len800_90x \
  --num-samples 90 \
  --max-len 800 \
  --generation-chunk-size 30 \
  --prompt-preset debug_fixed_prompt_two_track \
  --no-base \
  --only-model rules_curriculum
