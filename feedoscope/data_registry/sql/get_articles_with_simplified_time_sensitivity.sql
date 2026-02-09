-- Get all articles that have a "0-urgency" or "1-urgency" user tag.
-- Used for training the distilled urgency ModernBERT model.
--
-- The tag value IS the training label:
--   "0-urgency" -> urgency_label = 0 (evergreen, still relevant in a year)
--   "1-urgency" -> urgency_label = 1 (time-sensitive, loses relevance quickly)
--
-- Tags are initially assigned by the LLM pipeline (make time_simple), but the
-- user can manually change them in Miniflux to correct misclassifications.
-- The pipeline never overwrites existing tags, so manual corrections are always
-- reflected here.
--
-- Returns both read and unread articles. The Python training script
-- (llm_learn_urgency.py) handles class balancing, prioritizing read articles
-- (whose tags are more trustworthy since the user has seen them).
--
-- The e.status column is included so the training script can distinguish
-- read vs unread articles during class balancing.
select
    e.id as article_id,
    e.title,
    e.starred,
    f.title as feed_name,
    e.content,
    e.url as link,
    e.author,
    e.published_at as date_entered,
    e.changed_at as last_read,
    null as time_sensitivity_score,
    COALESCE(e.tags, array[]::text[]) as tags,
    e.vote,
    e.status,
    CASE ut.title
        WHEN '0-urgency' THEN 0
        WHEN '1-urgency' THEN 1
    END as urgency_label
from
    entries e
    join feeds f on e.feed_id = f.id
    -- INNER JOIN: only articles that have an urgency tag are included.
    -- Articles without any urgency tag are excluded from training data.
    join entry_user_tags eut on eut.entry_id = e.id
    join user_tags ut on ut.id = eut.user_tag_id
        and ut.user_id = 1
        and ut.title in ('0-urgency', '1-urgency')
order by
    e.id asc;
