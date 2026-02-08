-- Get all articles that have a simplified time sensitivity score.
-- Used for training the distilled urgency ModernBERT model.
-- Returns article data joined with the binary urgency label.
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
    tss.score as urgency_label
from
    entries e
    join feeds f on e.feed_id = f.id
    join time_sensitivity_simplified tss on tss.article_id = e.id
order by
    e.id asc;
