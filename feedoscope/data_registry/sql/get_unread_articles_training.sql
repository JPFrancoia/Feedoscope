-- Get unread articles for training (unlabeled data)
-- In Miniflux: status='unread' AND vote != -1
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
    e.vote
from
    entries e
    join feeds f on e.feed_id = f.id
    left join time_sensitivity ts on ts.article_id = e.id
where
    e.status = 'unread'
    and e.vote != -1  -- Exclude bad articles
    and e.starred = false
    and e.published_at > now() - interval '1 year'
order by
    e.id asc
limit 2000;
