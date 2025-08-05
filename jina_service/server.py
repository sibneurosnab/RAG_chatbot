import gc
import logging
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModel
from typing import Optional
from chunker import LateChunker

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JinaServer")

# Глобальные переменные для модели и токенизатора
model: Optional[PreTrainedModel] = None
tokenizer: Optional[PreTrainedTokenizer] = None
chunker_instance: Optional[LateChunker] = None

# Конфигурация
MODEL_NAME = "jinaai/jina-embeddings-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_TOKENS = 8192
RATE_LIMIT = "60/minute"

# Инициализация Flask приложения
app = Flask(__name__)

# Настройка лимитера запросов
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
    storage_uri="memory://",
)

# Функция для загрузки модели
def load_model_and_tokenizer():
    global model, tokenizer, chunker_instance

    logger.info("Загрузка модели и токенизатора...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True)

    if torch.cuda.is_available():
        _model = _model.to(DEVICE)
        logger.info(f"Модель загружена на GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("GPU не доступен, используется CPU")

    _model.eval()
    
    # Присваиваем глобальным переменным до инициализации чанкера
    model = _model
    tokenizer = _tokenizer
    
    # Инициализация чанкера с загруженными моделью и токенизатором
    chunker_instance = LateChunker(model=model, tokenizer=tokenizer)

    logger.info("Модель, токенизатор и чанкер успешно инициализированы")
    if hasattr(chunker_instance, 'task_supported'):
        logger.info(f"Поддержка task/адаптеров через adapter_mask: {'Да' if chunker_instance.task_supported else 'Нет'}")
    
    # Логируем доступные task, если они есть
    if hasattr(model, '_adaptation_map'):
        logger.info(f"Доступные task в модели: {list(model._adaptation_map.keys())}")

# Вспомогательные функции
@torch.inference_mode()
def _gc():
    """Очистка памяти."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

# Эндпоинты-заглушки
@app.route("/embed", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def embed():
    logger.warning("Эндпоинт /embed не предназначен для Late Chunking. Используйте /embed_late_chunk.")
    return jsonify({"error": "/embed is not for Late Chunking. Use /embed_late_chunk"}), 400

@app.route("/embed_tokens", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def embed_tokens():
    logger.error("Эндпоинт /embed_tokens временно не доступен.")
    return jsonify({"error": "/embed_tokens temporarily unavailable"}), 501

@app.route("/embed_mean", methods=["POST"])
@limiter.limit(RATE_LIMIT)
def embed_mean():
    logger.error("Эндпоинт /embed_mean временно не доступен.")
    return jsonify({"error": "/embed_mean temporarily unavailable"}), 501

# Эндпоинт для Late Chunking
@app.route("/embed_late_chunk", methods=["POST"])
@limiter.limit("30 per minute")
def embed_late_chunk():
    logger.info(f"Запрос на /embed_late_chunk с телом (первые 200 симв): {str(request.data)[:200]}")
    try:
        if not request.is_json:
            logger.warning(f"Неверный Content-Type для /embed_late_chunk: {request.content_type}")
            return jsonify({"error": "Content-Type должен быть application/json"}), 415
        
        data = request.get_json()
        if not data or not isinstance(data, dict):
            logger.warning("Неверный формат данных JSON для /embed_late_chunk.")
            return jsonify({"error": "Неверный формат данных JSON"}), 400

        texts_to_process = data.get("texts", [])
        if not isinstance(texts_to_process, list):
            logger.warning(f"Поле 'texts' не является списком для /embed_late_chunk: {type(texts_to_process)}")
            return jsonify({"error": "Поле 'texts' должно быть списком"}), 400
        if not texts_to_process:
            logger.info("Получен пустой список текстов для /embed_late_chunk.")
            return jsonify({"results": []}), 200

        task_param = data.get("task", "retrieval.passage")
        strategy_param = data.get("strategy", "paragraph")
        chunk_size_param = data.get("chunk_size")
        overlap_param = data.get("overlap")

        logger.info(f"Параметры для /embed_late_chunk: strategy={strategy_param}, chunk_size={chunk_size_param}, overlap={overlap_param}, task={task_param}, num_texts={len(texts_to_process)}")

        if not chunker_instance:
             logger.error("Чанкер не инициализирован!")
             return jsonify({"error": "Сервис (чанкер) не инициализирован"}), 503

        # Проверяем поддержку task
        task_supported = getattr(chunker_instance, 'task_supported', False)
        if task_param and not task_supported:
            logger.warning(f"Запрошен task={task_param}, но модель не поддерживает adapter_mask. Эмбеддинги будут созданы без адаптации.")
        elif task_param and task_supported:
            # Проверяем наличие task в adaptation_map
            if hasattr(model, '_adaptation_map') and task_param in model._adaptation_map:
                logger.info(f"Task {task_param} найден в adaptation_map модели (ID: {model._adaptation_map[task_param]})")
            else:
                logger.warning(f"Task {task_param} не найден в adaptation_map модели")

        results = []
        for i, text_content in enumerate(texts_to_process):
            if not isinstance(text_content, str) or not text_content.strip():
                logger.warning(f"Пропущен пустой или некорректный текст на позиции {i} для /embed_late_chunk.")
                results.append({
                    "embeddings": [], 
                    "texts": [], 
                    "num_chunks": 0, 
                    "original_text_index": i, 
                    "error": "Empty or invalid text",
                    "task_supported": task_supported,
                    "task_used": None
                })
                continue
            
            logger.info(f"Обработка текста {i+1}/{len(texts_to_process)} (длина: {len(text_content)} симв.) для /embed_late_chunk")
            try:
                chunk_data = chunker_instance.encode_with_chunks(
                    text_content,
                    strategy=strategy_param,
                    chunk_size=chunk_size_param, 
                    overlap=overlap_param,
                    task=task_param
                )
                logger.info(f"Для текста {i+1} (late_chunk) получено {chunk_data.get('num_chunks', 0)} чанков.")
                if chunk_data.get("embeddings") and len(chunk_data.get("embeddings")) > 0:
                     logger.info(f"  Форма первого эмбеддинга для текста {i+1} (late_chunk): {np.shape(chunk_data['embeddings'][0])}")
                
                # Добавляем информацию о статусе task, если её нет в ответе чанкера
                if 'task_supported' not in chunk_data:
                    chunk_data['task_supported'] = task_supported
                if 'task_used' not in chunk_data:
                    chunk_data['task_used'] = task_param if task_supported else None
                
                # Добавляем индекс исходного текста
                chunk_data['original_text_index'] = i
                
                results.append(chunk_data)
            except Exception as e_inner:
                logger.error(f"Ошибка при обработке текста {i+1} (late_chunk): {e_inner}", exc_info=True)
                results.append({
                    "embeddings": [], 
                    "texts": [], 
                    "num_chunks": 0, 
                    "original_text_index": i, 
                    "error": str(e_inner),
                    "task_supported": task_supported,
                    "task_used": None
                })

        _gc()
        
        logger.info(f"Запрос /embed_late_chunk успешно обработан. Возвращено результатов для {len(results)} текстов.")
        return jsonify({
            "results": results,
            "task_supported": task_supported
        })

    except Exception as e:
        logger.error(f"Критическая ошибка в эндпоинте /embed_late_chunk: {e}", exc_info=True)
        return jsonify({"error": f"Внутренняя ошибка сервера: {str(e)}"}), 500

# Загрузка модели при запуске
if __name__ == "__main__":
    load_model_and_tokenizer()
    app.run(host="0.0.0.0", port=8008, debug=False)
else:
    load_model_and_tokenizer()
