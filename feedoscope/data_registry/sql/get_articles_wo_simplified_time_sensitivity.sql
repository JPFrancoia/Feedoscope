-- Get articles from the last 6 months that don't yet have a simplified time sensitivity score.
-- Used to build training data for the distilled urgency model.
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
    e.vote
from
    entries e
    join feeds f on e.feed_id = f.id
where
    e.published_at > now() - interval '6 months'
    and not exists (
        select 1 from time_sensitivity_simplified tss
        where tss.article_id = e.id
    )
order by
    e.id asc;
