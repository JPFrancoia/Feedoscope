select e.id as entry_id, e.title, coalesce(e.content, '') as content
from entries e
where e.published_at >= %(until)s - %(days)s * interval '1 day'
    and e.published_at <= %(until)s
    and e.status <> 'removed'
    and not exists (select 1 from entries_importance i where i.entry_id = e.id)
order by e.id;
