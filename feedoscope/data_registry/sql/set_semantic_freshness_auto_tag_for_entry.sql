with removed as (
    delete from entry_user_tags eut
    using user_tags ut
    where eut.entry_id = %(entry_id)s
      and eut.user_tag_id = ut.id
      and ut.user_id = 1
      and ut.title in (
          'lt-24h-auto-freshness', '1-3d-auto-freshness', '4-7d-auto-freshness',
          '8-30d-auto-freshness', '1-6m-auto-freshness', 'evergreen-auto-freshness'
      )
)
insert into entry_user_tags (entry_id, user_tag_id)
values (%(entry_id)s, %(user_tag_id)s)
on conflict do nothing;
