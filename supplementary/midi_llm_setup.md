# MIDI-LLM

В проекте настроен отдельный контур для `MIDI-LLM` как практичной
текстовой symbolic music модели, которой можно передавать `ABC` или
другое MIDI-представление в текстовом виде и задавать judge-вопрос.

По умолчанию setup рассчитан на публичный checkpoint:

- `slseanwu/MIDI-LLM_Llama-3.2-1B`

## Инициализация

```bash
./run.sh init-midi-llm
```

Это:

- создаст `models/midi_llm/.venv`
- установит зависимости
- скачает модель в `models/midi_llm/checkpoints/model`

Без скачивания весов:

```bash
./run.sh init-midi-llm --skip-model-download
```

С другим HF repo:

```bash
./run.sh init-midi-llm --model-id <repo_id>
```

## Inference

Если у вас уже есть `ABC` или другой symbolic text:

```bash
./run.sh midi-llm-infer \
  --input-file path/to/example.abc \
  --question "Evaluate the musical quality of the melody-harmony relationship. Return Score and Reason."
```

Если нужно добавить исходный generation prompt:

```bash
./run.sh midi-llm-infer \
  --input-file path/to/example.abc \
  --context-prompt-file path/to/prompt.txt \
  --question "Evaluate the musical quality of the melody-harmony relationship. Return Score and Reason."
```

С сохранением JSON:

```bash
./run.sh midi-llm-infer \
  --input-file path/to/example.abc \
  --question "Evaluate the musical quality of the melody-harmony relationship. Return Score and Reason." \
  --output-json outputs/midi_llm/result.json
```

## Важно

Этот контур работает именно с **текстовым symbolic input**, а не с
картинкой. То есть сюда логично подавать:

- `ABC`
- compact event sequence
- другое сериализованное MIDI-представление

Если позже понадобится массовый `judge-eval` по набору примеров, его
лучше строить уже поверх этого runner отдельно.
