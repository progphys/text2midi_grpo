#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEFAULT_OUT="$PROJECT_ROOT/outputs/archives/text2midi_len1200_datasphere_bundle.tar.gz"
OUT_PATH="${1:-$DEFAULT_OUT}"

mkdir -p "$(dirname "$OUT_PATH")"

TMP_EXCLUDES=$(mktemp)
cat > "$TMP_EXCLUDES" <<'EOF'
.git
.git/*
.venv
.venv/*
models
models/*
outputs
outputs/*
wandb
wandb/*
.env
*.tar.gz
__pycache__
__pycache__/*
*.pyc
*.pyo
*.safetensors
*.pt
*.bin
EOF

tar \
  --exclude-from="$TMP_EXCLUDES" \
  -czf "$OUT_PATH" \
  -C "$(dirname "$PROJECT_ROOT")" \
  "$(basename "$PROJECT_ROOT")"

rm -f "$TMP_EXCLUDES"
echo "$OUT_PATH"
