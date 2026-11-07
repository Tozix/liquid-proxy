# liquid-proxy

OpenAI-совместимый API-шлюз над бесплатным [Liquid AI Playground](https://playground.liquid.ai/).
Принимает запросы в формате OpenAI, сам регистрирует анонимные сессии на playground.liquid.ai,
проксирует чат и возвращает ответы в OpenAI-формате — стримом (SSE) и одним вызовом.

Один бинарник, ноль зависимостей (чистый Go stdlib).

## Запуск

```bash
cd liquid-proxy
go build -o liquid-proxy .
PORT=8080 ./liquid-proxy
```

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PORT` | `8080` | порт HTTP-сервера |
| `API_KEY` | *(пусто — открыт)* | если задан, требует `Authorization: Bearer <key>` или `x-api-key: <key>` |
| `UPSTREAM` | `https://playground.liquid.ai` | адрес playground |
| `POOL_MAX` | `8` | максимум анонимных сессий (каждая = отдельный анонимный аккаунт) |
| `PREWARM` | `2` | сколько сессий создать при старте (0 = лениво по запросу) |

## Эндпоинты

- `POST /v1/chat/completions` — OpenAI-совместимый чат (`stream: true|false`)
- `GET /v1/models` — список моделей
- `GET /healthz` — статус пула сессий

## Модели

| ID | upstream slug | тип |
|---|---|---|
| `lfm2.5-24b-a2b` | `lfm2.5-24b-a2b-20261027` | text (флагман, reasoning) |
| `lfm3-1.7b` | `lfm3-1.7b-20261027` | text |
| `lfm3-vl-2b` | `lfm3-vl-2b-20261027` | vision (картинки прокси не проксирует) |
| `lfm2.5-2.6b` | `lfm2.5-2.6b-20260811` | text |
| `lfm2.5-vl-3b` | `lfm2.5-vl-3b-20260811` | vision |
| `lfm2.5-1.2b-jp` | `lfm2.5-1.2b-jp-202606-20260604` | text, японский |

Также принимаются полные slug'и. Неизвестное имя модели → fallback на `lfm2.5-24b-a2b` (+ warning в лог).

## Примеры

```bash
# одиночный вызов
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-24b-a2b","messages":[{"role":"user","content":"Привет!"}]}'

# стрим
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm3-1.7b","stream":true,"messages":[{"role":"user","content":"Считай до 5"}]}'
```

Python / Node с официальным SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="anything")
r = client.chat.completions.create(model="lfm2.5-24b-a2b",
    messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content)
```

## Особенности поведения

- **Reasoning**: `<think>…</think>` модели отдаётся в поле `delta.reasoning_content`
  (формат DeepSeek; понимают OpenWebUI, Cherry Studio, LibreChat и т.п.), ответ — в `content`.
- **Мультичент**: OpenAI-клиент присылает всю историю — прокси сериализует её в один промпт
  (`system` как есть, остальные с префиксами `User:` / `Assistant:`) и создаёт новый чат на каждый запрос.
- **usage**: `prompt_tokens`/`completion_tokens` — оценка (байты/4), upstream реальные токены не отдаёт
  (только `tokensPerSecond`). `stream_options.include_usage` поддержан.
- **Сессии**: пул анонимных аккаунтов с exclusive checkout. Причина — upstream ротирует
  session-токен и инвалидирует предыдущий, параллельное использование одной сессии ломает её.
  401/403 от upstream → сессия утилизируется и запрос один раз ретраится на новой сессии
  (до отправки первого байта клиенту, т.к. авторизация проверяется на этапе заголовков).
- **Картинки (vision)**: поддержаны для `lfm3-vl-2b` и `lfm2.5-vl-3b` в стандартном OpenAI-формате
  (`content`-массив с `image_url`): data URI (`data:image/png;base64,…`) и http(s)-URL (прокси скачает сам).
  Картинка загружается в storage playground'а (presign → S3 PUT, как в вебе) и передаётся через `imageUuids`.
  Не-vision модель с картинкой → 400 с подсказкой. Лимиты: ≤4 картинки на запрос, ≤10 МБ каждая.
  Voice-модели не поддержаны.
- **Tool calling (function calling)**: `tools` из OpenAI-запроса прокидываются в промпт, нативные
  вызовы модели (`<|tool_call_start|>[name(arg='v')]<|tool_call_end|>`) парсятся и конвертируются в
  OpenAI `delta.tool_calls` / `message.tool_calls` с `finish_reason: "tool_calls"` (стрим и non-stream).
  История `assistant.tool_calls` + `role:"tool"` результаты тоже сериализуются. Аргументы —
  Python-литералы модели (строки/числа/bool/None/списки/словари) конвертируются в JSON; экзотика
  уходит как `{"_raw": "..."}`. Работает с OpenAI SDK и агентскими клиентами (проверено циклом
  tool call → tool result → финальный ответ).

## Лимиты

**На стороне прокси** (настраивается):

| Лимит | Значение | Источник |
|---|---|---|
| Одновременные запросы | `POOL_MAX` сессий (дефолт 8); лишние ждут в очереди | пул сессий |
| Тело запроса | 8 МБ | `io.LimitReader` в обработчике |
| Контекст прокси не режет | проксируется как есть | — |

**На стороне upstream** (эмпирика от 2026-11-07, `lfm2.5-24b-a2b`):

| Контекст промпта | Результат |
|---|---|
| ~31k токенов | ✅ работает, точный recall («последнее слово» найдено) |
| ~91k токенов | ✅ отвечает формально |
| ~139k токенов | ⚠️ деградация: reasoning зацикливается, content пустой |
| ~208k токенов | ❌ тихий отказ: HTTP 200, но пустой ответ (1 токен) |

Практический потолок ≈ **90–100k токенов** (оценка байты/4; реальный токенайзер может отличаться).
Переполнение НЕ возвращает ошибку — приходят пустые ответы, обрабатывайте это на клиенте.
Параллельность: 4 одновременных запроса через разные сессии прошли без 429; явных
rate-limit заголовков у upstream нет.

Малые модели (`lfm3-1.7b`) иногда выдают пустой ответ на простой запрос — просто повторите.

## Проверено (2026-11-07)

- `go vet`, `go test` — зелёные (включая табличные тесты сплиттера на chunk-границах)
- curl: non-stream + SSE, `/v1/models`, `/healthz`
- официальный `openai` SDK (Node): non-stream, stream c `include_usage`, мульти-turn
- auth-мидлварь: 401 без/с неверным ключом, 200 с ключом (Bearer и x-api-key)
- vision: red/blue PNG через data URI → «Red»/«Blue» (обе VL-модели), http URL → «Blue»,
  text-модель с картинкой → 400 с подсказкой

## Сервис на macOS (launchd)

Чтобы прокси жил фоном, стартовал при входе в систему и перезапускался после падений:

```bash
go build -o liquid-proxy .   # бинарник должен лежать по пути из plist
```

Создай `~/Library/LaunchAgents/com.tozix.liquid-proxy.plist` (подставь свои пути — `ProgramArguments`
указывает на бинарник, env-переменные настраиваются в блоке `EnvironmentVariables`), затем:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tozix.liquid-proxy.plist   # установить+запустить
launchctl kickstart -k gui/$(id -u)/com.tozix.liquid-proxy                              # перезапустить (после go build)
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.tozix.liquid-proxy.plist      # остановить и снять
rm ~/Library/LaunchAgents/com.tozix.liquid-proxy.plist                                  # удалить совсем
```

`RunAtLoad` — старт при логине, `KeepAlive` — автоперезапуск при падении, логи — в `StandardOutPath`.

## Дисклеймер

Автоматизация playground идёт вразрез с ToS Liquid AI; это инструмент для личного
использования/исследований. Прокси старается не мусорить: пул сессий ограничен, а каждый
чат после ответа удаляется на upstream (`DELETE /api/chats/{id}`, best-effort — лог `[cleanup]`).
Загруженные картинки в их storage не удаляются (API удаления нет).
