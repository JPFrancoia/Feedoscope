-- Get unsummarized Miniflux entries from a recent publication window.
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
where
    e.published_at > now() - interval '1 day' * %(days)s
    and e.one_line_summary = ''
order by
    e.published_at desc,
    e.id desc
limit
    %(limit)s;
