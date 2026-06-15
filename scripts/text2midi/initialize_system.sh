#!/usr/bin/env bash
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_DIR="$PROJECT_ROOT/models/text2midi"

resolve_python_bin() {
    local candidate resolved
    for candidate in "${TEXT2MIDI_PYTHON:-}" python3.10 python3 python; do
        [ -n "$candidate" ] || continue
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        if resolved=$("$candidate" -c 'import sys; print(sys.executable)' 2>/dev/null); then
            echo "$resolved"
            return 0
        fi
    done
    return 1
}

if ! PYTHON_BIN="$(resolve_python_bin)"; then
    echo "Python 3.10+ is required but not found."
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Mode: system Python, no venv"

# 1. Clone external Text2MIDI repo.
if [ ! -d "$MODEL_DIR/.git" ]; then
    echo "Cloning amaai-lab/text2midi..."
    mkdir -p "$PROJECT_ROOT/models"
    git clone https://github.com/amaai-lab/text2midi "$MODEL_DIR"
else
    echo "models/text2midi already exists, skipping clone."
fi

# 2. Install dependencies into the currently selected Python environment.
echo "Installing requirements into system/current Python environment..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$MODEL_DIR/requirements.txt"
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"

# 3. Download weights from HuggingFace.
echo "Downloading model weights..."
"$PYTHON_BIN" - <<PYEOF
import os
from huggingface_hub import hf_hub_download, snapshot_download

root = "$PROJECT_ROOT"
weights_dir = os.path.join(root, "models/text2midi/weight_models/text2midi")
tok_dir = os.path.join(root, "models/text2midi/weight_models/flan-t5-base")

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(tok_dir, exist_ok=True)

print("Downloading pytorch_model.bin and vocab_remi.pkl if missing...")
for fname in ["pytorch_model.bin", "vocab_remi.pkl"]:
    target = os.path.join(weights_dir, fname)
    if os.path.exists(target):
        print(f"  {fname} already present.")
        continue
    hf_hub_download(repo_id="amaai-lab/text2midi", filename=fname, local_dir=weights_dir)
    print(f"  downloaded {fname}")

print("Downloading flan-t5-base tokenizer if missing...")
if not os.path.exists(os.path.join(tok_dir, "tokenizer_config.json")):
    snapshot_download(
        repo_id="google/flan-t5-base",
        local_dir=tok_dir,
        ignore_patterns=["*.bin", "*.safetensors", "flax_model*", "tf_model*", "rust_model*"],
    )
else:
    print("  tokenizer already present.")
print("All weights downloaded!")
PYEOF

echo ""
echo "Done! Try:"
echo "  ./run.sh train critic_rank_only_ffn_15rollouts --synthetic --max-steps 3 --save-every 1 --no-wandb"
