# Зона ML — пайплайн «До репоста»

Всё, что относится к LLM-пайплайну, схемам, проверке результата и тестам.
HTTP API, поиск и получение страниц — в `backend/`; frontend — отдельно.

## Состав

- `doreposta/schemas.py` — схемы `Source`, `Claim`, `ClaimResult`, `EvaluationResult`, ошибки
- `doreposta/prompts.py` — промпты извлечения и оценки утверждений
- `doreposta/llm_client.py` — `LLMClient` (реальная модель) и `FakeLLMClient` (тесты)
- `doreposta/pipeline.py` — `extract_claims`, `evaluate_claims`, `compose_reply`, `validate_result`, `summarize`
- `doreposta/checks.py` — детерминированные проверки: годы, числа, отрицания, обрезка, отбор источников
- `doreposta/context_build.py` — сборка блока материалов с лимитом на источник (4000 символов, до 6 источников)
- `doreposta/recording.py` — `RecordingLLMClient` / `ReplayLLMClient` для воспроизводимых прогонов
- `tests/` — сценарии пайплайна, правила статусов, ужесточения (`test_hardening.py`), набор live-кейсов
- `example.py` — сквозной пример на fake-клиенте (без ключа)
- `live_run.py` — прогон на живой модели

## Запуск

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r ml/requirements.txt

cd ml
../.venv/Scripts/python.exe -m pytest -q      # тесты без API-ключа
../.venv/Scripts/python.exe example.py        # демонстрация пайплайна
../.venv/Scripts/python.exe live_run.py       # только с LLM_API_KEY
```

Из корня репозитория те же тесты запускаются как `pytest -q` (`ml/conftest.py`
добавляет `ml/` в `sys.path` и запускает async-тесты без плагина).

Переменные окружения: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT`,
`LLM_RETRIES` (повторы на 429/5xx, по умолчанию 2), `LLM_BACKOFF` (пауза, 0.5 с)
(шаблоны — `ml/.env.example` и `.env.example` в корне). `LLM_BASE_URL` указывается
без `/v1` — клиент сам дописывает `/v1/chat/completions`.

## Согласование промптов и проверок

`prompts.py` и `validate_result` — одна пара: модель отвечает, проверка перепроверяет
ответ детерминированно. Меняя промпт, учитывай, что уже проверяется в коде:

- `supported` требует evidence с `relation=supports`, `contradicted` — с `contradicts`,
  `context` — с `supports`/`adds_context` и названным условием в `explanation`;
  иначе статус понижается до `insufficient`.
- Год из утверждения должен быть в цитате или в `published_at`; расхождение чисел
  (когда числа есть с обеих сторон) и отрицаний тоже понижает статус.
- `content_kind="snippet"` не даёт `supported`/`contradicted`/`context`.
- Цитаты короче двух значимых слов отбрасываются, одинаковые утверждения схлопываются.
- Материалы собирает `context_build.sources_block`: до 6 источников, до 4000 символов
  на источник, обрезка помечается в `limitations`.

Правила закреплены в `tests/test_status_rules.py` и `tests/test_hardening.py` —
меняешь правило, правь тест рядом, иначе он покажет расходление.

## Пайплайн «Лакмус» (`ml/lakmus`)

Новый контракт ТЗ (разделы 3, 6.2–6.9): мультимодальный чекер
image / text / url с оценкой доверия, «двумя адвокатами» и
рекомендациями следующих шагов.

### Структура

- `lakmus/schemas.py` — единые схемы: `CheckResult`, `Trust`, `SignalResult`, `Debate`, `NextStep`.
- `lakmus/context.py` — `CheckContext`, `SearchHit`, `SearchProvider`.
- `lakmus/runner.py` — `run_signals`: параллельный запуск с таймаутами и SSE.
- `lakmus/fusion.py` — слияние сигналов по Йосангу, уровни доверия.
- `lakmus/debate.py` — «два адвоката» и формирование `next_steps`.
- `lakmus/signals/` — сигналы: `metadata`, `ai_artifacts`, `manipulation`, `source_search`,
  `screenshot` (image); `claims`, `sources`, `quotes`, `rhetoric`, `machine_text` (text/url).
- `lakmus/pipeline.py` — оркестратор `check(ctx)`.
- `lakmus/metrics.py`, `dataset.py`, `demo.py` — метрики, набор и демо.
- `tools/eval.py`, `tools/export_schema.py` — оценка на разметке и JSON Schema контракта.
- `tests/test_pipeline_lakmus.py`, `test_signals_image.py`, `test_signals_text.py`,
  `test_debate.py`, `test_metrics.py` — тесты.

### Запуск

```bash
cd <корень репозитория>
py -3 -m pytest ml -q
py -3 -m ml.tools.eval
py -3 -m ml.tools.export_schema
```

### Пример использования

```python
import asyncio
from lakmus import CheckInput, InputType
from lakmus.context import CheckContext
from lakmus.pipeline import check

async def main():
    ctx = CheckContext(
        check_id="demo_1",
        input=CheckInput(type=InputType.text, extracted_text="Сегодня в Москве снег."),
        text="Сегодня в Москве снег.",
    )
    result = await check(ctx)
    print(result.trust.score, result.trust.level)

asyncio.run(main())
```

Контракт экспортируется в `contracts/check_result.schema.json`.
- `lakmus/llm.py` — `RecordingLLMClient` / `ReplayLLMClient` для кэширования и воспроизведения вызовов LLM.
- `lakmus/search.py` — `BackendSearchProvider` (поиск через backend endpoint) и `FakeSearchProvider` (для тестов).
- `lakmus/validate.py` — `validate_check_result` для проверки `CheckResult` по `contracts/check_result.schema.json`.
