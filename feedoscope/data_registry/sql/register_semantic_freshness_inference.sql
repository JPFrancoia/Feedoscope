insert into semantic_freshness_inference (
    article_id,
    model_key,
    bucket_probabilities,
    expected_lifetime_days,
    last_updated
)
values (
    %(article_id)s,
    %(model_key)s,
    %(bucket_probabilities)s,
    %(expected_lifetime_days)s,
    now()
)
on conflict (article_id, model_key) do update
set
    bucket_probabilities = excluded.bucket_probabilities,
    expected_lifetime_days = excluded.expected_lifetime_days,
    last_updated = excluded.last_updated;
