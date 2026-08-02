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
    e.status
from entries e
join feeds f on e.feed_id = f.id
left join time_sensitivity ts on ts.article_id = e.id
where e.id > %(after_article_id)s
order by e.id asc
limit %(batch_size)s;
