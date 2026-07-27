select e.id as article_id, e.title
from entries e
join entry_user_tags eut on eut.entry_id = e.id
join user_tags ut on ut.id = eut.user_tag_id
where e.status = 'read'
  and ut.user_id = 1
  and ut.title in (
      'lt-24h-freshness', '1-3d-freshness', '4-7d-freshness',
      '8-30d-freshness', '1-6m-freshness', 'evergreen-freshness'
  )
group by e.id, e.title
having count(*) > 1
order by e.id asc;
