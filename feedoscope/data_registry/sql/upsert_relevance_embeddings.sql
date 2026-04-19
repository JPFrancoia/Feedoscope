insert into relevance_embeddings (
    article_id,
    model_name,
    max_length,
    text_prep_mode,
    prep_version,
    text_hash,
    embedding
)
values (
    %(article_id)s,
    %(model_name)s,
    %(max_length)s,
    %(text_prep_mode)s,
    %(prep_version)s,
    %(text_hash)s,
    %(embedding)s
)
on conflict (
    article_id,
    model_name,
    max_length,
    text_prep_mode,
    prep_version
)
do update set
    text_hash = excluded.text_hash,
    embedding = excluded.embedding,
    last_updated = now();
