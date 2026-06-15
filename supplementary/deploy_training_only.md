# Text2MIDI Training-Only Bundle

Этот архив предназначен только для запуска обучения.

Внутри нет:

- `ChatMusician`
- setup для `ChatMusician`
- compare/infer/probe скриптов
- локальных outputs/wandb
- внешней модели `models/text2midi`
- `.venv`

## 1. Распаковать

```bash
tar -xzf text2midi_training_only_bundle.tar.gz
cd Text2midi
```

## 2. Инициализация

Обычный серверный режим:

```bash
./run.sh init
```

Notebook/DataSphere режим:

```bash
./run.sh init-system
```

Это:

- создаст окружение
- скачает `amaai-lab/text2midi`
- скачает базовые веса
- поставит зависимости

## 3. Основной запуск

Прямое дообучение FFN без LoRA:

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --max-steps 1000 \
  --save-every 10
```

Что делает этот эксперимент:

- reward только от `final_observer`
- GRPO
- прямое обучение только `linear1` и `linear2`
- `num_rollouts=18`
- `generation_chunk_size=18`
- `update_mini_batch=18`
- `rollout_max_len=1200`
- `lr=1e-5`
- warmup `13`

## 4. Короткий smoke test

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --max-steps 3 \
  --save-every 1 \
  --no-wandb
```

## 5. Продолжение обучения

```bash
./run.sh train final_observer_only_ffn_finetune_18rollouts \
  --synthetic \
  --resume-latest \
  --max-steps 1500 \
  --save-every 10
```

## 6. Куда пишутся чекпоинты

```bash
outputs/checkpoints/final_observer_only_ffn_finetune_18rollouts/
```
