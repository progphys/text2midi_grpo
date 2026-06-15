#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LLAMA_MIDI_DIR="$PROJECT_ROOT/models/llama_midi"
LLAMA_MIDI_VENV="$LLAMA_MIDI_DIR/.venv"
LLAMA_MIDI_MODEL_ID="${LLAMA_MIDI_MODEL_ID:-dx2102/llama-midi}"
DOWNLOAD_MODEL=1

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-model-download)
            DOWNLOAD_MODEL=0
            shift
            ;;
        --model-id)
            LLAMA_MIDI_MODEL_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-model-download] [--model-id <hf_repo_or_path>]"
            exit 1
            ;;
    esac
done

resolve_python_bin() {
    local candidate resolved
    for candidate in "${TEXT2MIDI_PYTHON:-}" "$PROJECT_ROOT/.venv/bin/python" python3.11 python3.10 python3.9 python3 python; do
        [ -n "$candidate" ] || continue
        if [ -x "$candidate" ]; then
            if resolved=$("$candidate" -c 'import sys, venv; assert (3, 9) <= sys.version_info[:2] <= (3, 11); print(sys.executable)' 2>/dev/null); then
                echo "$resolved"
                return 0
            fi
            continue
        fi
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        if resolved=$("$candidate" -c 'import sys, venv; assert (3, 9) <= sys.version_info[:2] <= (3, 11); print(sys.executable)' 2>/dev/null); then
            echo "$resolved"
            return 0
        fi
    done
    if command -v pyenv >/dev/null 2>&1; then
        for candidate in 3.11.9 3.10.19 3.10.14 text2midi-3.10; do
            resolved="$(pyenv which "$candidate" 2>/dev/null || true)"
            if [ -x "$resolved" ] && "$resolved" -c 'import sys, venv; assert (3, 9) <= sys.version_info[:2] <= (3, 11)' >/dev/null 2>&1; then
                echo "$resolved"
                return 0
            fi
            resolved="$(pyenv prefix "$candidate" 2>/dev/null)/bin/python"
            if [ -x "$resolved" ] && "$resolved" -c 'import sys, venv; assert (3, 9) <= sys.version_info[:2] <= (3, 11)' >/dev/null 2>&1; then
                echo "$resolved"
                return 0
            fi
        done
    fi
    return 1
}

if ! PYTHON_BIN="$(resolve_python_bin)"; then
    echo "Python 3.9-3.11 with venv support is required for llama-midi but not found."
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "llama-midi model id: $LLAMA_MIDI_MODEL_ID"

mkdir -p "$LLAMA_MIDI_DIR/checkpoints"

if [ ! -d "$LLAMA_MIDI_VENV" ]; then
    echo "Creating llama-midi venv at $LLAMA_MIDI_VENV ..."
    "$PYTHON_BIN" -m venv "$LLAMA_MIDI_VENV"
else
    echo "llama-midi venv already exists, skipping creation."
fi

LLAMA_MIDI_PIP="$LLAMA_MIDI_VENV/bin/pip"
LLAMA_MIDI_PYTHON="$LLAMA_MIDI_VENV/bin/python"

echo "Installing llama-midi Python requirements..."
"$LLAMA_MIDI_PIP" install --upgrade pip setuptools wheel
"$LLAMA_MIDI_PIP" install -r "$PROJECT_ROOT/requirements-llama-midi.txt"

if [ "$DOWNLOAD_MODEL" -eq 1 ]; then
    echo "Downloading llama-midi snapshot..."
    "$LLAMA_MIDI_PYTHON" - <<PYEOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$LLAMA_MIDI_MODEL_ID",
    local_dir="$LLAMA_MIDI_DIR/checkpoints/model",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("llama-midi model snapshot is ready.")
PYEOF
else
    echo "Skipping model download."
fi

echo ""
echo "Done! llama-midi environment:"
echo "  source models/llama_midi/.venv/bin/activate"
echo ""
echo "Inference example:"
echo "  ./run.sh llama-midi-infer --input-file example.txt --question \"Describe this MIDI-derived music text.\""
