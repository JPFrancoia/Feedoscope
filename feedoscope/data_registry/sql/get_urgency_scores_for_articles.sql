-- Fetch cached urgency scores for a set of articles.
-- Used by main.py to look up urgency probabilities for time-decay calculation.
select
    article_id,
    urgency_score
from
    urgency_inference
where
    article_id = any(%(article_ids)s);
