with candidates as (
    select
        e.id as entry_id,
        replace(min(ut.title), '-auto-freshness', '-freshness') as reviewed_title
    from entries e
    join entry_user_tags eut on eut.entry_id = e.id
    join user_tags ut on ut.id = eut.user_tag_id
    where e.status = 'read'
      and ut.user_id = 1
      and ut.title in (
          'lt-24h-auto-freshness', '1-3d-auto-freshness', '4-7d-auto-freshness',
          '8-30d-auto-freshness', '1-6m-auto-freshness', 'evergreen-auto-freshness'
      )
      and not exists (
          select 1
          from entry_user_tags reviewed_eut
          join user_tags reviewed_ut on reviewed_ut.id = reviewed_eut.user_tag_id
          where reviewed_eut.entry_id = e.id
            and reviewed_ut.user_id = 1
            and reviewed_ut.title in (
                'lt-24h-freshness', '1-3d-freshness', '4-7d-freshness',
                '8-30d-freshness', '1-6m-freshness', 'evergreen-freshness'
            )
      )
    group by e.id
    having count(*) = 1
), removed as (
    delete from entry_user_tags eut
    using candidates c
    where eut.entry_id = c.entry_id
      and eut.user_tag_id = (
          select id from user_tags
          where user_id = 1
            and title = replace(c.reviewed_title, '-freshness', '-auto-freshness')
      )
    returning c.entry_id, c.reviewed_title
)
insert into entry_user_tags (entry_id, user_tag_id)
select removed.entry_id, ut.id
from removed
join user_tags ut on ut.user_id = 1 and ut.title = removed.reviewed_title
on conflict do nothing;
