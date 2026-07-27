insert into semantic_freshness_teacher_labels (
    article_id,
    horizon,
    confidence,
    source
)
values (
    %(article_id)s,
    %(horizon)s,
    %(confidence)s::confidence,
    %(source)s
)
on conflict (article_id) do update
set
    horizon = excluded.horizon,
    confidence = excluded.confidence,
    source = excluded.source;
