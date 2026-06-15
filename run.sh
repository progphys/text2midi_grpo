#!/usr/bin/env bash
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
if [ -n "${TEXT2MIDI_PYTHON:-}" ]; then
    PYTHON="$TEXT2MIDI_PYTHON"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON="$(command -v python3.10)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON=""
fi

cd "$PROJECT_ROOT"

# ── Цвета ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

usage() {
    echo -e "${BOLD}Использование:${NC}"
    echo -e "  ${CYAN}./run.sh init${NC}                          — первоначальная настройка (venv + веса)"
    echo -e "  ${CYAN}./run.sh init-system${NC}                   — настройка в системный Python (для Notebook/DataSphere)"
    echo -e "  ${CYAN}./run.sh init-midi-llm${NC}                 — отдельная настройка MIDI-LLM"
    echo -e "  ${CYAN}./run.sh init-llama-midi${NC}               — отдельная настройка llama-midi"
    echo -e "  ${CYAN}./run.sh train <эксп> [--synthetic]${NC}    — обучить эксперимент"
    echo -e "  ${CYAN}./run.sh infer --caption \"...\"${NC}         — сгенерировать MIDI из текста"
    echo -e "  ${CYAN}./run.sh compare --experiment rules_plus_critic${NC} — base vs finetuned"
    echo -e "  ${CYAN}./run.sh midi-llm-infer --input-file ... --question \"...\"${NC} — reasoning по ABC/MIDI-тексту"
    echo -e "  ${CYAN}./run.sh llama-midi-infer --input-file ... --question \"...\"${NC} — reasoning через llama-midi"
    echo -e "  ${CYAN}./run.sh grok-midi-judge --midi ... --output-dir ...${NC} — внешний Grok-судья по MIDI→ABC"
    echo -e "  ${CYAN}./run.sh baseline${NC}                      — оценить оригинальную модель"
    echo -e "  ${CYAN}./run.sh eval  <эксп> <чекпоинт>${NC}       — оценить чекпоинт после обучения"
    echo -e ""
    echo -e "${BOLD}Эксперименты:${NC}"
    echo -e "  symbolic_rules     Только rule-based rewards"
    echo -e "  rules_triplet      Только 3 rules: key_conditioned + meter + tempo_bin_tolerant"
    echo -e "  rules_triplet_ffn_only  Тот же rules_triplet, но LoRA только на linear1/linear2"
    echo -e "  key_meter_plus_critic  2 rules: key_conditioned + meter + fixed-pairs critic"
    echo -e "  key_meter_plus_critic_ffn_only  То же, но LoRA только на linear1/linear2"
    echo -e "  key_meter_plus_critic_ffn_only_lr1e4  То же, но FFN-only и lr=1e-4"
    echo -e "  key_meter_plus_critic_lr1e4  Attention+FFN LoRA и lr=1e-4"
    echo -e "  critic_rank_only   Только fixed-pairs critic rank reward"
    echo -e "  critic_rank_only_ffn_15rollouts  Только critic, FFN-only LoRA, 15 rollout'ов"
    echo -e "  final_observer_only_ffn_18rollouts_stable  Только final_observer, FFN-only LoRA, 18 rollout'ов"
    echo -e "  final_observer_only_ffn_fixedprompt_25rollouts_len800  Только final_observer, 1 prompt, FFN-only LoRA, 25 rollout'ов, len=800"
    echo -e "  final_observer_only_ffn_fixedprompt_30rollouts_len800  Только final_observer, 1 prompt, FFN-only LoRA, 30 rollout'ов, len=800"
    echo -e "  final_observer_only_ffn_fixedprompt_30rollouts_len800_lr3e7  Только final_observer, 1 prompt, 30 rollout'ов, len=800, lr=3e-7"
    echo -e "  final_observer_only_ffn_fixedprompt_35rollouts_len800  Только final_observer, 1 prompt, FFN-only LoRA, 35 rollout'ов, len=800, lr=3e-7"
    echo -e "  final_observer_only_attn_lora_18rollouts  Только final_observer, attention-only LoRA, 18 rollout'ов"
    echo -e "  final_observer_only_attn_lora_30rollouts  Только final_observer, attention-only LoRA, 30 rollout'ов"
    echo -e "  final_observer_only_ffn_finetune_18rollouts  Только final_observer, прямое дообучение FFN linear1/linear2"
    echo -e "  final_observer_only_attn_finetune_30rollouts  Только final_observer, прямое дообучение attention-only, 30 rollout'ов"
    echo -e "  final_observer_only_last2_attn_ffn_finetune_30rollouts  Только final_observer, прямое дообучение последних 2 слоев attention+FFN"
    echo -e "  final_observer_only_attn_ffn_finetune_30rollouts  Только final_observer, прямое дообучение attention+FFN, 30 rollout'ов"
    echo -e "  critic_rank_only_attn_15rollouts_lr7e5  Только critic, attention-only LoRA, 15 rollout'ов, lr=7e-5"
    echo -e "  critic_rank_only_attn_ffn_15rollouts  Только critic, attention+FFN LoRA, 15 rollout'ов"
    echo -e "  rules_plus_critic  Rules + fixed-pairs critic rank ${YELLOW}← основной${NC}"
    echo -e "  key_plus_critic_lora  Два rewards: key_conditioned + fixed-pairs critic"
    echo -e ""
    echo -e "${BOLD}Примеры:${NC}"
    echo -e "  ./run.sh init"
    echo -e "  ./run.sh init-midi-llm"
    echo -e "  ./run.sh init-llama-midi"
    echo -e "  ./run.sh train key_plus_critic_lora --synthetic"
    echo -e "  ./run.sh train rules_plus_critic --synthetic"
    echo -e "  ./run.sh train critic_rank_only_ffn_15rollouts --synthetic"
    echo -e "  ./run.sh train final_observer_only_ffn_18rollouts_stable --synthetic"
    echo -e "  ./run.sh train final_observer_only_ffn_fixedprompt_25rollouts_len800 --synthetic"
    echo -e "  ./run.sh train final_observer_only_ffn_fixedprompt_30rollouts_len800 --synthetic"
    echo -e "  ./run.sh train final_observer_only_ffn_fixedprompt_30rollouts_len800_lr3e7 --synthetic"
    echo -e "  ./run.sh train final_observer_only_ffn_fixedprompt_35rollouts_len800 --synthetic"
    echo -e "  ./run.sh train final_observer_only_attn_lora_18rollouts --synthetic"
    echo -e "  ./run.sh train final_observer_only_attn_lora_30rollouts --synthetic"
    echo -e "  ./run.sh train final_observer_only_ffn_finetune_18rollouts --synthetic"
    echo -e "  ./run.sh train final_observer_only_attn_finetune_30rollouts --synthetic"
    echo -e "  ./run.sh train final_observer_only_last2_attn_ffn_finetune_30rollouts --synthetic"
    echo -e "  ./run.sh train final_observer_only_attn_ffn_finetune_30rollouts --synthetic"
    echo -e "  ./run.sh train critic_rank_only_attn_15rollouts_lr7e5 --synthetic"
    echo -e "  ./run.sh train critic_rank_only_attn_ffn_15rollouts --synthetic"
    echo -e "  ./run.sh infer --caption \"A nostalgic folk tune in D minor\""
    echo -e "  ./run.sh compare --experiment rules_plus_critic --num-prompts 10"
    echo -e "  ./run.sh midi-llm-infer --input-file example.abc --question \"Evaluate the melody-harmony relationship.\""
    echo -e "  ./run.sh llama-midi-infer --input-file example.txt --question \"Describe this MIDI-derived music text.\""
    echo -e "  ./run.sh grok-midi-judge --input-root outputs/judge_eval/base_vs_final_50 --output-dir outputs/grok_judge/base_vs_final_50 --limit 5 --prepare-only"
    echo -e "  ./run.sh baseline"
    echo -e "  ./run.sh eval rules_plus_critic outputs/checkpoints/rules_plus_critic/step_00200"
}

check_venv() {
    if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
        echo -e "${RED}Ошибка: Python не найден. Сначала запусти: ./run.sh init или ./run.sh init-system${NC}"
        exit 1
    fi
}

# ── Команды ───────────────────────────────────────────────────────────────────

cmd_init() {
    echo -e "${GREEN}▶ Инициализация...${NC}"
    bash "$PROJECT_ROOT/scripts/text2midi/initialize.sh"
}

cmd_init_system() {
    echo -e "${GREEN}▶ Инициализация в системный Python...${NC}"
    bash "$PROJECT_ROOT/scripts/text2midi/initialize_system.sh"
}

cmd_init_midi_llm() {
    echo -e "${GREEN}▶ Инициализация MIDI-LLM...${NC}"
    bash "$PROJECT_ROOT/scripts/text2midi/initialize_midi_llm.sh" "$@"
}

cmd_init_llama_midi() {
    echo -e "${GREEN}▶ Инициализация llama-midi...${NC}"
    bash "$PROJECT_ROOT/scripts/text2midi/initialize_llama_midi.sh" "$@"
}

cmd_infer() {
    check_venv
    echo -e "${GREEN}▶ Генерация MIDI...${NC}"
    "$PYTHON" "$PROJECT_ROOT/scripts/infer.py" "$@"
}

cmd_baseline() {
    check_venv
    echo -e "${GREEN}▶ Оценка оригинальной модели (baseline)...${NC}"
    "$PYTHON" "$PROJECT_ROOT/scripts/eval.py" --baseline
}

cmd_compare() {
    check_venv
    echo -e "${GREEN}▶ Сравнительный inference: base vs finetuned...${NC}"
    "$PYTHON" "$PROJECT_ROOT/scripts/compare_infer.py" "$@"
}

cmd_midi_llm_infer() {
    local midi_llm_python="$PROJECT_ROOT/models/midi_llm/.venv/bin/python"
    if [ ! -x "$midi_llm_python" ]; then
        echo -e "${RED}MIDI-LLM не инициализирован. Сначала запусти: ./run.sh init-midi-llm${NC}"
        exit 1
    fi
    echo -e "${GREEN}▶ Inference MIDI-LLM...${NC}"
    "$midi_llm_python" "$PROJECT_ROOT/tools/midi_llm/infer_midi_llm.py" "$@"
}

cmd_llama_midi_infer() {
    local llama_midi_python="$PROJECT_ROOT/models/llama_midi/.venv/bin/python"
    if [ ! -x "$llama_midi_python" ]; then
        echo -e "${RED}llama-midi не инициализирован. Сначала запусти: ./run.sh init-llama-midi${NC}"
        exit 1
    fi
    echo -e "${GREEN}▶ Inference llama-midi...${NC}"
    "$llama_midi_python" "$PROJECT_ROOT/tools/llama_midi/infer_llama_midi.py" "$@"
}

cmd_grok_midi_judge() {
    check_venv
    echo -e "${GREEN}▶ Grok judge: MIDI → ABC → score...${NC}"
    "$PYTHON" "$PROJECT_ROOT/tools/grok_judge/grok_midi_judge.py" "$@"
}

cmd_train() {
    check_venv
    local exp="$1"; shift
    if [ -z "$exp" ]; then
        echo -e "${RED}Укажи эксперимент: symbolic_rules, critic_rank_only, rules_plus_critic${NC}"
        exit 1
    fi
    echo -e "${GREEN}▶ Обучение эксперимента ${BOLD}$exp${NC}${GREEN}...${NC}"
    "$PYTHON" "$PROJECT_ROOT/scripts/train.py" --experiment "$exp" "$@"
}

cmd_eval() {
    check_venv
    local exp="$1"
    local ckpt="$2"
    if [ -z "$exp" ] || [ -z "$ckpt" ]; then
        echo -e "${RED}Укажи эксперимент и путь к чекпоинту${NC}"
        echo -e "Пример: ./run.sh eval rules_plus_critic outputs/checkpoints/rules_plus_critic/step_00200"
        exit 1
    fi
    # Извлекаем номер шага из имени папки (step_02000 → 2000)
    local step=$(basename "$ckpt" | grep -o '[0-9]*' | tail -1)
    echo -e "${GREEN}▶ Оценка ${BOLD}$exp${NC}${GREEN} / чекпоинт ${BOLD}$ckpt${NC}${GREEN} (step=$step)...${NC}"
    "$PYTHON" "$PROJECT_ROOT/scripts/eval.py" \
        --experiment "$exp" \
        --checkpoint "$ckpt" \
        --step "$step"
}

# ── Роутинг ───────────────────────────────────────────────────────────────────

case "$1" in
    init)     shift; cmd_init "$@" ;;
    init-system) shift; cmd_init_system "$@" ;;
    init-midi-llm) shift; cmd_init_midi_llm "$@" ;;
    init-llama-midi) shift; cmd_init_llama_midi "$@" ;;
    infer)    shift; cmd_infer "$@" ;;
    compare)  shift; cmd_compare "$@" ;;
    midi-llm-infer) shift; cmd_midi_llm_infer "$@" ;;
    llama-midi-infer) shift; cmd_llama_midi_infer "$@" ;;
    grok-midi-judge) shift; cmd_grok_midi_judge "$@" ;;
    baseline) shift; cmd_baseline "$@" ;;
    train)    shift; cmd_train "$@" ;;
    eval)     shift; cmd_eval "$@" ;;
    *)        usage ;;
esac
