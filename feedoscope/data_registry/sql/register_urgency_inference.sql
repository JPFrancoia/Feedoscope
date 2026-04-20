insert into urgency_inference (
    article_id,
    model_key,
    urgency_score
)
values (
    %(article_id)s,
    %(model_key)s,
    %(urgency_score)s
)
on conflict (article_id, model_key)
do update set
    urgency_score = excluded.urgency_score,
    last_updated = now();
