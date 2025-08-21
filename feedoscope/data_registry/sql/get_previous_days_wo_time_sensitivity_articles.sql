select
    e.id as article_id,
    e.title,
    e.content
from
    ttrss_entries as e
    left join time_sensitivity ts on e.id = ts.article_id
where
    ts.article_id is null
    and e.date_entered >= now() - interval '1 day' * %(number_of_days)s
order by
    e.id asc;
