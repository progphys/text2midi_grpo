#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MIDI_LLM_DIR="$PROJECT_ROOT/models/midi_llm"
MIDI_LLM_VENV="$MIDI_LLM_DIR/.venv"
MIDI_LLM_MODEL_ID="${MIDI_LLM_MODEL_ID:-slseanwu/MIDI-LLM_Llama-3.2-1B}"
DOWNLOAD_MODEL=1

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-model-download)
            DOWNLOAD_MODEL=0
            shift
            ;;
        --model-id)
            MIDI_LLM_MODEL_ID="$2"
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
    echo "Python 3.9-3.11 with venv support is required for MIDI-LLM but not found."
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "MIDI-LLM model id: $MIDI_LLM_MODEL_ID"

mkdir -p "$MIDI_LLM_DIR/checkpoints"

if [ ! -d "$MIDI_LLM_VENV" ]; then
    echo "Creating MIDI-LLM venv at $MIDI_LLM_VENV ..."
    "$PYTHON_BIN" -m venv "$MIDI_LLM_VENV"
else
    echo "MIDI-LLM venv already exists, skipping creation."
fi

MIDI_LLM_PIP="$MIDI_LLM_VENV/bin/pip"
MIDI_LLM_PYTHON="$MIDI_LLM_VENV/bin/python"

echo "Installing MIDI-LLM Python requirements..."
"$MIDI_LLM_PIP" install --upgrade pip setuptools wheel
"$MIDI_LLM_PIP" install -r "$PROJECT_ROOT/requirements-midi-llm.txt"

if [ "$DOWNLOAD_MODEL" -eq 1 ]; then
    echo "Downloading MIDI-LLM snapshot..."
    "$MIDI_LLM_PYTHON" - <<PYEOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$MIDI_LLM_MODEL_ID",
    local_dir="$MIDI_LLM_DIR/checkpoints/model",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("MIDI-LLM model snapshot is ready.")
PYEOF
else
    echo "Skipping model download."
fi

echo ""
echo "Done! MIDI-LLM environment:"
echo "  source models/midi_llm/.venv/bin/activate"
echo ""
echo "Inference example:"
echo "  ./run.sh midi-llm-infer --input-file example.abc --question \"Evaluate the melody-harmony relationship.\""
