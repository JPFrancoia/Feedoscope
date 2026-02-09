-- Get all articles that have a 0-urgency or 1-urgency user tag.
-- Used for training the distilled urgency ModernBERT model.
-- The tag value IS the training label (no time_sensitivity_simplified fallback).
-- Returns both read and unread articles; class balancing is done in Python.
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
    join entry_user_tags eut on eut.entry_id = e.id
    join user_tags ut on ut.id = eut.user_tag_id
        and ut.user_id = 1
        and ut.title in ('0-urgency', '1-urgency')
order by
    e.id asc;
