#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT_NAME="${WANDB_PROJECT:-text2midi-rules}"
WANDB_RUN_NAME="${WANDB_NAME:-track_select_rules3_rejection_lr5e6_a100}"

echo "==> Initializing Text2MIDI project environment"
./run.sh init

echo "==> Starting training"
WANDB_PROJECT="$WANDB_PROJECT_NAME" \
WANDB_NAME="$WANDB_RUN_NAME" \
./run.sh train track_select_rules3_attn_lora_5rollouts_a100 --synthetic
