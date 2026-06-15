#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="outputs/fixed_prompt_ab/fixed_prompt_top1_len800_90x"

run_step() {
  local script="$1"
  local expected_total="$2"

  echo "==> Running ${script}"
  "./tools/fixed_prompt_ab/run_models/${script}"

  local current_total
  current_total="$(find "${RUN_ROOT}" -name '*.mid' | wc -l)"
  echo "==> MIDI saved: ${current_total} / ${expected_total}"
}

run_step gen_rules_fixed.sh 180
run_step gen_rules_curriculum.sh 270
run_step gen_critic_only_final_final.sh 360
run_step gen_critic_after_curriculum.sh 450

echo "==> Done. Final MIDI count:"
find "${RUN_ROOT}" -name '*.mid' | wc -l

