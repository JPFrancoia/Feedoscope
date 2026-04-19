select
    article_id,
    text_hash,
    embedding
from
    relevance_embeddings
where
    article_id = any(%(article_ids)s)
    and model_name = %(model_name)s
    and max_length = %(max_length)s
    and text_prep_mode = %(text_prep_mode)s
    and prep_version = %(prep_version)s;
