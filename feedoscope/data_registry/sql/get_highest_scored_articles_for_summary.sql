-- Get the highest-scored Miniflux entries for one-line summary generation.
select
    e.id as article_id,
    e.title,
    e.starred,
    f.title as feed_name,
    coalesce(e.content, '') as content,
    e.url as link,
    coalesce(e.author, '') as author,
    e.published_at as date_entered,
    e.changed_at as last_read,
    coalesce(e.tags, array[]::text[]) as tags,
    e.vote,
    e.status
from
    entries e
    join feeds f on e.feed_id = f.id
order by
    e.score desc,
    e.id desc
limit
    %(limit)s;
