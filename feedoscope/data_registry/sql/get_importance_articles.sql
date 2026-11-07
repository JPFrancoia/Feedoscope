select id as entry_id, title, coalesce(content, '') as content
from entries
where id = any(%(entry_ids)s) and status <> 'removed'
order by id;
