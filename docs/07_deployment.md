# 7. Развёртывание проекта

> Документ описывает **пошаговый** процесс установки и запуска системы RAG-бота на собственной машине или сервере. Все действия проверены на Ubuntu 22.04 LTS и Windows 11 (WSL 2).  
> При использовании других дистрибутивов отличия минимальны – путь к драйверам и менеджеру пакетов.

---

## 7.1 Предварительные требования

1. **Аппаратное обеспечение**  
   • CPU x86-64 с AVX           – достаточно для пилотного запуска.  
   • **GPU NVIDIA** с ≥ 6 ГБ VRAM – **желательно** для Whisper / Jina-моделей.  
   • Свободное место ≥ 15 ГБ (образы + кэш моделей).
2. **ОС / Kernel**  
   • Linux (рекомендуется) или Windows 11 + WSL 2.  
   • Для GPU – драйвер NVIDIA ≥ 535 и `nvidia-container-toolkit`.
3. **ПО**  
   • Git ≥ 2.30  
   • Docker ≥ 24  
   • Docker Compose v2 (входит в новые Docker Desktop / Engine).
4. **Аккаунты / токены**  
   • Supabase – проект + сервис-key + URL  
   • Telegram Bot Token  
   • OpenRouter API-key (или замените на GPT-совместимый сервис)  
   • Firecrawl / DumplingAI токены *(опционально, только при использовании paid-SaaS узлов)*.

> **Без GPU**: сервисы `jina-embed`/`jina-reranker` и `whisper-asr` работают в FP32 на CPU – медленнее, но без доработок Docker-файлов.

---

## 7.2 Клонирование репозитория

```bash
# SSH или HTTPS – без разницы
$ git clone https://github.com/<YOUR_USERNAME>/rag-telegram-latechunk.git
$ cd rag-telegram-latechunk
```

---

## 7.3 Настройка переменных окружения

1. Скопируйте пример:

```bash
$ cp .env.example .env
```

2. Заполните обязательные поля в `.env` (пустые – оставить **нельзя**):

| Переменная | Пример | Назначение |
|------------|--------|------------|
| `N8N_HOST` | `chat.example.com` | Публичный домен/IPv4 сервера. |
| `WEBHOOK_URL` | `https://chat.example.com` | Внешний URL веб-хуков Telegram. |
| `SUPABASE_URL` | `https://xxxxx.supabase.co` | Endpoint вашего проекта. |
| `SUPABASE_SERVICE_KEY` | `eyJ...` | `service_role` ключ (полный доступ). |
| `TG_TOKEN` | `123456:ABC-DEF...` | Токен бота.| 
| `OPENROUTER_API_KEY` | `sk-...` | Ключ OpenRouter. |

Дополнительные переменные описаны в комментариях `.env.example`.

*При локальном тесте* можно указать `N8N_HOST=localhost` и раскомментировать блок `ports:` в `deploy/docker-compose.yml` → n8n станет доступен по `http://localhost:5678`.

---

## 7.4 Импорт схемы Supabase

```bash
# один раз после создания проекта Supabase
$ psql "${SUPABASE_URL}" -f supabase/01_schema.sql 
# либо скопируйте содержимое файла в SQL-редактор UI и нажмите Run
```

Файл `supabase/01_schema.sql` содержит:
• создание таблиц `documents_paragraphs`, `chat_messages`, `user_state`;  
• расширение `pgvector` и HNSW-индекс;  
• функции `search_documents` и `insert_and_trim_pairs` с нужными правами.

> **Важно:** выполняйте скрипт строго под ролью `service_role`, иначе не получится создать функции и выдать права.

---

## 7.5 Сборка и запуск контейнеров

```bash
# Сборка (первый запуск ~10-15 минут из-за скачивания моделей)
$ docker compose -f deploy/docker-compose.yml up --build -d

# Проверка статуса
$ docker compose ps
```

Если установлен `watch`, можно следить за логами:

```bash
$ docker compose logs -f --tail=50 jina-embed-gpu
```

> Признак готовности – строка `* Running on http://0.0.0.0:8008` в логе `jina-embed-gpu`.

### 7.5.1 Запуск без GPU

Удалите или закомментируйте секции `deploy.resources` и переменную `NVIDIA_VISIBLE_DEVICES` в следующих сервисах:

* `jina-embed-gpu`
* `jina-reranker-gpu`
* `whisper-asr`

После этого пересоберите образы – они автоматически переключатся на CPU-режим FP32.

---

## 7.6 Проверка работоспособности

1. Перейдите в браузере: `http://<HOST_ИЛИ_IP>:5678` – откроется UI n8n.  
2. Запустите *первый* workflow **`RAG CHATBOT Main`** вручную и убедитесь, что статус `idle`.  
3. Откройте Telegram-чат с ботом → `/start` → выберите режим **База** и отправьте PDF/URL.  
   *Бот должен ответить «Внесено».  В БД Supabase появятся строки в таблице `documents_paragraphs`.*
4. Переключитесь в режим **Чат** и задайте вопрос – бот вернёт ответ + блок ground-source.

---

## 7.7 Частые ошибки и их решение

| Симптом | Причина | Решение |
|---------|---------|---------|
| `ERR_MODULE_NOT_FOUND: flash-attn` при сборке | Недостаточно VRAM / старая CUDA | Уберите пакет `flash-attn` из `requirements.txt` или снижайте версию CUDA до 11.8 |
| `CUDA driver not found` | Нет `nvidia-container-toolkit` | Установите toolkit, перезапустите Docker daemon |
| `TLS handshake error` при обращении к Supabase | Неверный `SUPABASE_SERVICE_KEY` | Проверьте ключ, права `service_role`|
| Бот молчит после `/start` | Telegram не может доставить веб-хук | Убедитесь, что `WEBHOOK_URL` публичен и порты 80/443 открыты |

---

## 7.8 Удаление

```bash
$ docker compose down -v --remove-orphans
```

---

## 7.9 Статус и планы

* Разработка ведётся **public-first**; pull-requests приветствуются.  
* В backlog – автоматический импорт SQL → Supabase, CPU-friendly конфигурация и CI GitHub Actions для сборки образов.

---

**Готово!** После выполнения шагов бот полностью функционирует локально и может быть развёрнут на VPS/домашнем сервере.
