select e.id as article_id, e.title
from entries e
join entry_user_tags eut on eut.entry_id = e.id
join user_tags ut on ut.id = eut.user_tag_id
where e.status = 'read'
  and ut.user_id = 1
  and ut.title in (
      'fresh-lt-24h', 'fresh-1-3d', 'fresh-4-7d',
      'fresh-8-30d', 'fresh-1-6m', 'fresh-evergreen'
  )
group by e.id, e.title
having count(*) > 1
order by e.id asc;
