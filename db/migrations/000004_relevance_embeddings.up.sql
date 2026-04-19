create table relevance_embeddings (
    article_id bigint not null references entries(id) on delete cascade,
    model_name text not null,
    max_length integer not null,
    text_prep_mode text not null,
    prep_version integer not null,
    text_hash text not null,
    -- Store normalized float32 embeddings as raw bytes. This keeps the cache
    -- portable across environments where pgvector is not installed.
    embedding bytea not null,
    last_updated timestamp with time zone not null default now(),
    primary key (
        article_id,
        model_name,
        max_length,
        text_prep_mode,
        prep_version
    )
);
