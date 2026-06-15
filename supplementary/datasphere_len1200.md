# Text2MIDI DataSphere Bundle: `track_select_key_meter_note_density_len1200_a100`

Этот bundle предназначен для запуска эксперимента:

- `track_select_key_meter_note_density_len1200_a100`

Что внутри:

- весь необходимый код проекта
- `run.sh`
- `requirements.txt`
- скрипты `init` / `init-system`
- конфиг эксперимента `len1200`

Что специально не входит в архив:

- `.venv`
- `models/`
- `outputs/`
- `wandb/`
- `.git/`

Это сделано для того, чтобы архив был переносимым, а модель и зависимости скачались уже на DataSphere.

## Разворачивание на DataSphere

Распаковать архив:

```bash
tar -xzf text2midi_len1200_datasphere_bundle.tar.gz
cd Text2midi
```

Если `venv` в окружении создается нормально:

```bash
./run.sh init
```

Если это notebook/DataSphere окружение без `python3-venv`, использовать:

```bash
./run.sh init-system
```

Обе команды:

- клонируют `amaai-lab/text2midi` в `models/text2midi`
- устанавливают зависимости
- скачивают веса и tokenizer из Hugging Face

## Запуск обучения

Основной запуск:

```bash
./run.sh train track_select_key_meter_note_density_len1200_a100 --synthetic
```

Прямой запуск через Python:

```bash
./.venv/bin/python scripts/train.py \
  --experiment track_select_key_meter_note_density_len1200_a100 \
  --synthetic
```

## Что используется в этом конфиге

- `rollout_max_len: 1200`
- `key_profile_weight: 1.0`
- `meter_template_weight: 1.0`
- `note_density_weight: 0.2`
- attention-only LoRA: `to_qkv`, `to_out`
- `lr: 5e-6`
- `save_rollout_midis: true`

## Куда пишутся результаты

Чекпоинты:

```text
outputs/checkpoints/track_select_key_meter_note_density_len1200_a100/
```

MIDI rollout'ы:

```text
outputs/text2midi/track_select_key_meter_note_density_len1200_a100/
```

## W&B

Если ключ уже есть в окружении или в `.env`, запуск подхватит его автоматически.

Пример:

```bash
export WANDB_PROJECT=text2midi-grpo
./run.sh train track_select_key_meter_note_density_len1200_a100 --synthetic
```
