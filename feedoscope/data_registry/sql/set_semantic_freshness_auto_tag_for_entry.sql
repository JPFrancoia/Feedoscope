with removed as (
    delete from entry_user_tags eut
    using user_tags ut
    where eut.entry_id = %(entry_id)s
      and eut.user_tag_id = ut.id
      and ut.user_id = 1
      and ut.title in (
          'fresh-auto-lt-24h', 'fresh-auto-1-3d', 'fresh-auto-4-7d',
          'fresh-auto-8-30d', 'fresh-auto-1-6m', 'fresh-auto-evergreen'
      )
)
insert into entry_user_tags (entry_id, user_tag_id)
values (%(entry_id)s, %(user_tag_id)s)
on conflict do nothing;
