-- Get every unread article in one non-overlapping age range for controlled inference.
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
    and e.published_at <= now() - interval '1 day' * %(min_age_days)s
    and e.published_at > now() - interval '1 day' * %(max_age_days)s
order by
    e.id asc;
