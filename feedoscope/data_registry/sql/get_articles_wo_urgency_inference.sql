-- Get recent unread articles that don't yet have a cached urgency inference score.
-- Used by the distilled ModernBERT urgency model to infer only new articles.
select
    e.id as article_id,
    e.title,
    e.starred,
    e.score,
    f.title as feed_name,
    e.content,
    e.url as link,
    e.author,
    e.published_at as date_entered,
    e.changed_at as last_read,
    null as time_sensitivity_score,
    COALESCE(e.tags, array[]::text[]) as tags,
    e.vote,
    e.status
from
    entries e
    join feeds f on e.feed_id = f.id
where
    e.status = 'unread'
    and e.vote != -1
    and e.starred = false
    and e.published_at >= now() - interval '1 day' * %(number_of_days)s
    and not exists (
        select 1 from urgency_inference ui
        where ui.article_id = e.id
    )
order by
    e.id asc;
