-- =============================================================
--  Supabase initial schema for RAG-Telegram-LateChunk project
--  Run once per project:   psql <01_schema.sql>   или через UI
-- =============================================================

-- 1) Расширения ------------------------------------------------
create extension if not exists vector;

-- 2) Таблица документов ---------------------------------------
create table if not exists documents_paragraphs (
    id              bigserial primary key,
    type            text      not null,               -- web | video | file
    title           text,
    url             text,
    file_name       text,
    paragraph_text  text      not null,
    paragraph_idx   integer   not null,
    language        text,
    embedding       vector(768) not null,
    retrieved_at    timestamptz default now(),
    batch_id        text,
    idx_in_batch    integer,
    constraint uniq_ingest unique (batch_id, idx_in_batch)
);

--  HNSW-индекс для ANN-поиска
create index if not exists idx_embedding_hnsw
    on documents_paragraphs
 using hnsw
    (embedding vector_cosine_ops)
with (
    m               = 16,
    ef_construction = 200
);

-- 3) История чата ---------------------------------------------
create table if not exists chat_messages (
    id          bigserial primary key,
    user_id     text      not null,
    role        text      not null check (role in ('user','assistant')),
    message     text      not null,
    created_at  timestamptz default now()
);
create index if not exists idx_chat_user_created_at on chat_messages (user_id, created_at);

-- 4) Состояние пользователя -----------------------------------
create table if not exists user_state (
    user_id     text primary key,
    current_mode text not null,      -- chat | documents
    session_id   text,
    updated_at   timestamptz default now()
);

-- 5) Функция поиска -------------------------------------------
create or replace function search_documents(
    query_vector vector,
    top_k        integer default 15
)
returns table(
    id              bigint,
    paragraph_text  text,
    score           double precision,
    type            text,
    url             text,
    file_name       text,
    paragraph_idx   integer
) language plpgsql stable as $$
begin
    return query
    select dp.id,
           dp.paragraph_text,
           1 - (dp.embedding <=> query_vector) as score,
           dp.type,
           dp.url,
           dp.file_name,
           dp.paragraph_idx
    from   documents_paragraphs dp
    where  1 - (dp.embedding <=> query_vector) > 0.35  -- порог similarity
    order  by dp.embedding <=> query_vector
    limit  top_k;
end;$$;

grant execute on function search_documents(vector, integer) to anon, authenticated, service_role;

-- 6) Функция вставки + обрезки чата ---------------------------
create or replace function insert_and_trim_pairs(
    p_items      jsonb,
    p_keep_pairs integer default 15  -- хранить N пар
) returns void language plpgsql as $$
declare
  rec        jsonb;
  keep_count integer := p_keep_pairs * 2;  -- пар в 2 сообщения
  the_user   text;
begin
  -- 1) вставляем каждое сообщение
  for rec in select * from jsonb_array_elements(p_items) loop
    insert into chat_messages(user_id, message, role, created_at)
    values (
      rec->>'user_id',
      rec->>'message',
      rec->>'role',
      (rec->>'created_at')::timestamptz
    );
    the_user := rec->>'user_id';
  end loop;

  -- 2) обрезаем историю
  delete from chat_messages
  where user_id = the_user
    and id in (
      select id
      from chat_messages
      where user_id = the_user
      order by created_at desc
      offset keep_count
    );
end;$$;

grant execute on function insert_and_trim_pairs(jsonb, integer) to anon, authenticated, service_role;

-- =============================================================
--  Конец файла
-- =============================================================
