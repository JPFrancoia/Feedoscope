-- Get published articles (considered "bad")
-- In Miniflux: vote=-1 (equivalent to TTRSS published=true)
with numbered_articles as (
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
        ts.score as time_sensitivity_score,
        COALESCE(e.tags, array[]::text[]) as tags,
        e.vote,
        row_number() over (order by e.id asc) as rn
    from
        entries e
        join feeds f on e.feed_id = f.id
        left join time_sensitivity ts on ts.article_id = e.id
    where
        e.vote = -1  -- Bad articles
        and e.published_at > now() - interval '1 year'
    order by
        e.id asc
)
select
    article_id,
    title,
    starred,
    feed_name,
    content,
    link,
    author,
    date_entered,
    last_read,
    time_sensitivity_score,
    tags,
    vote
from
    numbered_articles
where
    rn > %(validation_size)s
order by
    article_id asc;
