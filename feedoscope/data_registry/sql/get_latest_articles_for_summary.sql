-- Get unsummarized entries among the newest Miniflux entries.
with selected_entries as (
    select
        e.id
    from
        entries e
    order by
        e.id desc
    limit
        %(limit)s
)
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
    join selected_entries s on e.id = s.id
    join feeds f on e.feed_id = f.id
where
    e.one_line_summary = ''
order by
    e.id desc;
