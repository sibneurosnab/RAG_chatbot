import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LateChunker:
    """
    Класс для реализации Late Chunking с поддержкой task через adapter_mask.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, strategy: str = "paragraph", chunk_size: int = 256, overlap: int = 0):
        """
        Инициализация чанкера.
        
        Args:
            model: Предобученная модель трансформера
            tokenizer: Токенизатор для модели
            strategy: Стратегия чанкирования ('paragraph', 'sentence', 'fixed')
            chunk_size: Размер чанка в токенах (для strategy='fixed')
            overlap: Перекрытие между чанками в токенах (для strategy='fixed')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Проверяем поддержку адаптеров в модели
        self.task_supported = hasattr(self.model, '_adaptation_map')
        if self.task_supported:
            logger.info("Модель поддерживает адаптеры через _adaptation_map")
        else:
            logger.warning("Модель не поддерживает адаптеры через _adaptation_map")
        
        self.paragraph_pattern = r'\n\s*\n'
        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')

    def _get_char_spans_for_chunks(
        self,
        text: str,
        strategy: str,
        chunk_size: int, # Используется для fixed
        overlap: int    # Используется для fixed
    ) -> List[Tuple[int, int, str]]: # Возвращает (start_char, end_char, chunk_text)
        """
        Разбивает текст на чанки и возвращает их СИМВОЛЬНЫЕ границы и текст.
        """
        if not text or not isinstance(text, str):
            logger.warning("Получен пустой или некорректный текст для _get_char_spans_for_chunks.")
            return []

        text_stripped = text.replace('\r\n', '\n').strip()
        char_spans_with_text = []
        raw_chunks_texts: List[str] = []

        if strategy == "paragraph":
            raw_chunks_texts = [chunk.strip() for chunk in re.split(self.paragraph_pattern, text_stripped) if chunk.strip()]
        elif strategy == "sentence":
            split_sentences = self.sentence_pattern.split(text_stripped)
            current_pos_in_stripped = 0
            for s in split_sentences:
                if not s.strip():
                    continue
                try:
                    original_s_pos = text_stripped.index(s.strip(), current_pos_in_stripped)
                    end_of_s_in_stripped = original_s_pos + len(s.strip())
                    full_s = s.strip()
                    if end_of_s_in_stripped < len(text_stripped) and text_stripped[end_of_s_in_stripped] in ".!?":
                        full_s += text_stripped[end_of_s_in_stripped]
                    raw_chunks_texts.append(full_s)
                    current_pos_in_stripped = original_s_pos + len(full_s)
                except ValueError:
                    raw_chunks_texts.append(s.strip())
                    current_pos_in_stripped += len(s) # Примерное смещение
            raw_chunks_texts = [chk for chk in raw_chunks_texts if chk]
        elif strategy == "fixed":
            # Для 'fixed' символьные границы не так важны на этом этапе,
            # т.к. нарезка будет по токенам. Вернем весь текст как один "чанк",
            # а _get_fixed_token_spans разберется с токенами.
            logger.debug("Стратегия 'fixed' обрабатывается в _get_fixed_token_spans, _get_char_spans_for_chunks не вызывается.")
            return [] 
        else:
            logger.error(f"Неизвестная стратегия чанкирования: {strategy}")
            raise ValueError(f"Неизвестная стратегия чанкирования: {strategy}")

        current_search_offset = 0
        for chunk_text_candidate in raw_chunks_texts:
            if not chunk_text_candidate: continue
            try:
                s_char = text.index(chunk_text_candidate, current_search_offset)
                e_char = s_char + len(chunk_text_candidate)
                char_spans_with_text.append((s_char, e_char, chunk_text_candidate))
                current_search_offset = e_char 
            except ValueError:
                logger.warning(f"Текст чанка '{chunk_text_candidate[:30]}...' не найден в исходном тексте при поиске символьных границ.")
        
        return char_spans_with_text

    def _map_char_spans_to_token_spans(
        self,
        char_spans_with_text: List[Tuple[int, int, str]],
        offset_mapping: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, str]]:
        token_spans_with_text = []
        for char_start, char_end, chunk_text_from_char_span in char_spans_with_text:
            start_token_idx = -1
            end_token_idx = -1

            for i, (offset_s, offset_e) in enumerate(offset_mapping):
                # Условие для первого токена чанка: его символьный диапазон перекрывается с char_start
                # и он не является спец. токеном (0,0), если это не самый первый токен (CLS)
                is_special_token_marker = (offset_s == 0 and offset_e == 0)

                if start_token_idx == -1:
                    if offset_e > char_start and offset_s < char_end : # Токен перекрывается с чанком
                         if not is_special_token_marker or i == 0 : # Либо не спец.токен, либо это CLS
                            start_token_idx = i
                
                # Условие для последнего токена чанка: его символьный диапазон перекрывается с char_end
                if start_token_idx != -1 : # Ищем конец только если нашли начало
                    if offset_s < char_end and offset_e >= char_start: # Токен все еще внутри или на границе чанка
                        end_token_idx = i + 1 # +1 для среза
                    
                    # Если токен начался за пределами char_end, значит предыдущий был последним
                    if offset_s >= char_end:
                        break
            
            if start_token_idx != -1 and end_token_idx != -1 and start_token_idx < end_token_idx:
                token_spans_with_text.append((start_token_idx, end_token_idx, chunk_text_from_char_span))
            else:
                logger.warning(f"Не удалось сопоставить токен-спан для символьного спана [{char_start}:{char_end}] ('{chunk_text_from_char_span[:30]}...'). Start: {start_token_idx}, End: {end_token_idx}")
        
        return token_spans_with_text
    
    def _get_fixed_token_spans(self, text: str, chunk_size: int, overlap: int,
                               offset_mapping_global: List[Tuple[int,int]],
                               input_ids_global_len: int) -> List[Tuple[int, int, str]]:
        token_spans_with_text = []

        if chunk_size <= 0:
            logger.error("chunk_size must be positive for fixed strategy")
            return []
        if chunk_size <= overlap:
            logger.error("chunk_size must be greater than overlap for fixed strategy")
            return []
        
        # Определяем индексы токенов, не являющихся специальными (например, CLS, SEP, PAD)
        # CLS обычно первый (индекс 0), SEP последний значащий.
        # Jina может не добавлять SEP в конце для 'passage'.
        # Мы будем ориентироваться на offset_mapping: (0,0) часто означает спец. токен.
        
        first_content_token_idx = 0
        # Ищем первый токен с ненулевым смещением (это начало контента после CLS, если CLS имеет (0,0))
        for idx, (os, oe) in enumerate(offset_mapping_global):
            if not(os == 0 and oe == 0):
                first_content_token_idx = idx
                break
        
        last_content_token_idx = input_ids_global_len
        # Ищем последний токен с ненулевым смещением с конца (это конец контента перед SEP/PAD)
        for idx in range(input_ids_global_len - 1, -1, -1):
            os, oe = offset_mapping_global[idx]
            if not(os == 0 and oe == 0):
                last_content_token_idx = idx + 1 # +1 так как это индекс для среза (exclusive)
                break
        
        num_effective_tokens = last_content_token_idx - first_content_token_idx

        if num_effective_tokens <= 0:
            return []

        for i in range(0, num_effective_tokens, chunk_size - overlap):
            # Индексы относительно начала контентных токенов
            start_in_content_slice = i
            end_in_content_slice = min(i + chunk_size, num_effective_tokens)
            
            # Абсолютные индексы в полной последовательности токенов (с CLS/SEP)
            actual_start_token = first_content_token_idx + start_in_content_slice
            actual_end_token = first_content_token_idx + end_in_content_slice
            
            if actual_start_token >= actual_end_token: # Пропускаем пустые или некорректные спаны
                continue

            # Получаем текст чанка по offset_mapping
            # Убедимся, что индексы не выходят за пределы offset_mapping
            if actual_start_token >= len(offset_mapping_global) or actual_end_token > len(offset_mapping_global):
                logger.warning(f"Индексы для fixed chunk {actual_start_token}-{actual_end_token} выходят за пределы offset_mapping длиной {len(offset_mapping_global)}")
                continue

            char_s = offset_mapping_global[actual_start_token][0]
            # Для char_e берем конец *последнего* токена в чанке.
            char_e = offset_mapping_global[actual_end_token - 1][1]
            
            chunk_text_from_offsets = text[char_s:char_e].strip()

            if not chunk_text_from_offsets:
                logger.debug(f"Пропущен пустой текстовый чанк для токенов {actual_start_token}-{actual_end_token}")
                continue
            
            token_spans_with_text.append((actual_start_token, actual_end_token, chunk_text_from_offsets))
            
            if end_in_content_slice == num_effective_tokens: # Достигли конца контентных токенов
                break
                
        return token_spans_with_text

    def _mean_pooling_with_spans(
        self,
        token_embeddings: torch.Tensor,      # (batch_size=1, seq_len, hidden_dim)
        attention_mask: torch.Tensor,      # (batch_size=1, seq_len)
        token_spans: List[Tuple[int, int, str]] 
    ) -> List[np.ndarray]:
        chunk_embeddings_list = []
        
        # Убираем батч-размерность, т.к. обрабатываем один документ за раз
        token_embeddings_unbatched = token_embeddings.squeeze(0) # (seq_len, hidden_dim)
        attention_mask_unbatched = attention_mask.squeeze(0)   # (seq_len)

        for start_idx, end_idx, _ in token_spans: # Текст чанка здесь не используется для пулинга
            if not (0 <= start_idx < end_idx <= token_embeddings_unbatched.shape[0]):
                logger.warning(f"Некорректный или выходящий за пределы спан [{start_idx}:{end_idx}] для token_embeddings формы {token_embeddings_unbatched.shape}")
                continue

            chunk_token_vectors = token_embeddings_unbatched[start_idx:end_idx]
            chunk_attention_mask = attention_mask_unbatched[start_idx:end_idx]

            if chunk_token_vectors.size(0) == 0:
                logger.warning(f"Пустой срез токен-векторов для спана [{start_idx}:{end_idx}]")
                continue

            expanded_mask = chunk_attention_mask.unsqueeze(-1).expand_as(chunk_token_vectors).float()
            sum_embeddings = torch.sum(chunk_token_vectors * expanded_mask, dim=0)
            sum_mask = torch.clamp(expanded_mask.sum(dim=0), min=1e-9) # Суммируем по оси токенов (dim=0)

            if sum_mask.eq(0).all(): # Если все элементы в sum_mask нулевые
                 logger.warning(f"Нулевая маска внимания для спана [{start_idx}:{end_idx}], возвращаем нулевой вектор.")
                 pooled = torch.zeros_like(sum_embeddings) # или torch.zeros_like(chunk_token_vectors[0])
            else:
                pooled = sum_embeddings / sum_mask
            
            normalized = F.normalize(pooled, p=2, dim=0)
            chunk_embeddings_list.append(normalized.cpu().numpy())
        
        return chunk_embeddings_list

    @torch.inference_mode()
    def encode_with_chunks(
        self,
        text: str,
        strategy: Optional[str] = None, 
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        task: Optional[str] = None
    ) -> Dict[str, Any]:
        
        current_strategy = strategy if strategy is not None else self.strategy
        current_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        current_overlap = overlap if overlap is not None else self.overlap

        logger.info(f"Начало Late Chunking для текста (первые 100 симв): {text[:100]}...")
        logger.info(f"Стратегия: {current_strategy}, chunk_size: {current_chunk_size}, overlap: {current_overlap}, task: {task}")

        if not text or not isinstance(text, str):
            logger.warning("Пустой или некорректный текст получен для encode_with_chunks.")
            return {"embeddings": [], "texts": [], "num_chunks": 0}

        # 1. Токенизируем весь текст ОДИН РАЗ
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=8192, 
            return_tensors="pt",
            add_special_tokens=True, 
            return_offsets_mapping=True 
        )
        # Важно: offset_mapping нужно до перемещения на device, если оно возвращает кортеж, а не тензор
        offset_mapping_cpu = encoding['offset_mapping'][0].cpu().numpy() # Для одного текста в батче
        encoding_on_device = {k: v.to(self.device) for k, v in encoding.items()}
        
        logger.info(f"Текст токенизирован. Input IDs shape: {encoding_on_device['input_ids'].shape}")
        input_ids_len = encoding_on_device['input_ids'].shape[1]

        # 2. Определяем токен-спаны для чанков
        token_spans_with_text: List[Tuple[int, int, str]] = []
        
        if current_strategy == "fixed":
            token_spans_with_text = self._get_fixed_token_spans(
                text, current_chunk_size, current_overlap, 
                offset_mapping_cpu, 
                input_ids_len
            )
        elif current_strategy in ["paragraph", "sentence"]:
            char_boundaries_with_text = self._get_char_spans_for_chunks(text, current_strategy, current_chunk_size, current_overlap)
            if not char_boundaries_with_text:
                 return {"embeddings": [], "texts": [], "num_chunks": 0}
            token_spans_with_text = self._map_char_spans_to_token_spans(
                char_boundaries_with_text,
                offset_mapping_cpu
            )
        else:
            logger.error(f"Неизвестная стратегия чанкирования: {current_strategy}")
            raise ValueError(f"Неизвестная стратегия чанкирования: {current_strategy}")

        final_token_spans_for_pooling = [(s, e) for s, e, _ in token_spans_with_text]
        chunk_texts_for_output = [txt for _, _, txt in token_spans_with_text]

        logger.info(f"Определено {len(final_token_spans_for_pooling)} токен-спанов для {len(chunk_texts_for_output)} текстовых чанков.")
        if not final_token_spans_for_pooling:
            return {"embeddings": [], "texts": [], "num_chunks": 0}

        # 3. Получаем "глобальные" токен-векторы
        model_input_dict = {k:v for k,v in encoding_on_device.items() if k!= 'offset_mapping'}
        
        # Используем adapter_mask для task, как в официальной реализации
        task_used = False
        if task and self.task_supported:
            try:
                # Проверяем наличие task в adaptation_map
                if hasattr(self.model, '_adaptation_map') and task in self.model._adaptation_map:
                    task_id = self.model._adaptation_map[task]
                    num_examples = model_input_dict['input_ids'].shape[0]
                    
                    # Создаем adapter_mask для всех примеров в батче
                    adapter_mask = torch.full(
                        (num_examples,), 
                        task_id, 
                        dtype=torch.int32, 
                        device=self.device
                    )
                    
                    # Добавляем adapter_mask в аргументы модели
                    model_input_dict['adapter_mask'] = adapter_mask
                    logger.info(f"Создан adapter_mask для task={task} (task_id={task_id})")
                    task_used = True
                else:
                    logger.warning(f"Task '{task}' не найден в adaptation_map модели")
            except Exception as e:
                logger.warning(f"Ошибка при создании adapter_mask для task={task}: {e}")
        
        # Вызываем модель с или без adapter_mask
        outputs = self.model(**model_input_dict, output_hidden_states=True)
        
        if hasattr(outputs, 'last_hidden_state'):
            global_token_embeddings = outputs.last_hidden_state # (batch_size=1, seq_len, hidden_dim)
        elif hasattr(outputs, 'hidden_states'):
            global_token_embeddings = outputs.hidden_states[-1]
        else:
            logger.error("Не удалось извлечь last_hidden_state из вывода модели.")
            return {"embeddings": [], "texts": [], "num_chunks": 0}
        
        attention_mask = encoding_on_device["attention_mask"] # (batch_size=1, seq_len)
        logger.info(f"Получены глобальные токен-эмбеддинги. Shape: {global_token_embeddings.shape}")

        # 4. Применяем pooling к глобальным токен-векторам для каждого чанка
        chunk_embeddings_list = self._mean_pooling_with_spans(
            global_token_embeddings,
            attention_mask,
            token_spans_with_text # Передаем спаны вместе с текстом, т.к. _mean_pooling_with_spans ожидает 3 элемента
        )
        logger.info(f"Получено {len(chunk_embeddings_list)} эмбеддингов чанков.")

        if chunk_embeddings_list:
            # np.array() здесь для логгирования формы, если chunk_embeddings_list[0] это ndarray
            first_emb_shape = chunk_embeddings_list[0].shape if isinstance(chunk_embeddings_list[0], np.ndarray) else np.array(chunk_embeddings_list[0]).shape
            logger.info(f"Форма первого эмбеддинга чанка: {first_emb_shape}")
        
        # Убедимся, что количество эмбеддингов соответствует количеству текстов
        # chunk_texts_for_output уже должен быть синхронизирован с token_spans_with_text,
        # а chunk_embeddings_list должен быть синхронизирован с token_spans_with_text (если _mean_pooling_with_spans не пропускает спаны)
        
        final_embeddings_for_json = []
        final_texts_for_json = []

        # Проверяем, что количество успешно созданных эмбеддингов
        # соответствует количеству текстов, для которых были определены валидные токен-спаны.
        # token_spans_with_text содержит только те чанки, для которых были найдены валидные токен-спаны.
        # _mean_pooling_with_spans возвращает эмбеддинги для этих же спанов.
        if len(chunk_embeddings_list) == len(token_spans_with_text):
            final_embeddings_for_json = [emb.tolist() for emb in chunk_embeddings_list]
            final_texts_for_json = [txt for _, _, txt in token_spans_with_text] # Берем тексты из token_spans_with_text
        else:
            logger.warning(f"Несоответствие количества эмбеддингов ({len(chunk_embeddings_list)}) и определенных токен-спанов ({len(token_spans_with_text)}). Это неожиданно.")
            # В качестве запасного варианта, берем по минимуму, хотя это указывает на проблему в логике выше
            min_len = min(len(chunk_embeddings_list), len(token_spans_with_text))
            if min_len > 0:
                 final_embeddings_for_json = [chunk_embeddings_list[i].tolist() for i in range(min_len)]
                 final_texts_for_json = [token_spans_with_text[i][2] for i in range(min_len)]

        return {
            "embeddings": final_embeddings_for_json, 
            "texts": final_texts_for_json,
            "num_chunks": len(final_embeddings_for_json),
            "task_supported": self.task_supported,
            "task_used": task if task_used else None
        }
