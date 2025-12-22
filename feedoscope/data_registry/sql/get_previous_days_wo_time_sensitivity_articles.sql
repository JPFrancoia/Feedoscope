-- Get articles from the previous X days without time sensitivity scores
-- In Miniflux: any status, no time_sensitivity entry
select
    e.id as article_id,
    e.title,
    e.starred,
    e.score,
    f.title as feed_name,
    e.content,
    e.url as link,
    e.author,
    e.created_at as date_entered,
    e.changed_at as last_read,
    null as time_sensitivity_score,
    COALESCE(e.tags, array[]::text[]) as tags,
    e.vote
from
    entries e
    join feeds f on e.feed_id = f.id
where
    e.created_at >= now() - interval '1 day' * %(number_of_days)s
    and not exists (
        select 1 from time_sensitivity ts
        where ts.article_id = e.id
    )
order by
    e.id asc;
