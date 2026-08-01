insert into super_important_inference (
    article_id,
    model_key,
    super_important_score
)
values (
    %(article_id)s,
    %(model_key)s,
    %(super_important_score)s
)
on conflict (article_id, model_key)
do update set
    super_important_score = excluded.super_important_score,
    last_updated = now();
