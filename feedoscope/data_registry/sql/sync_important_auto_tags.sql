-- Synchronize Feedoscope's automation-owned tag for articles processed in this run.
with important_entries as (
    select id, user_id
    from entries
    where id = any(%(important_article_ids)s)
),
important_tags as (
    insert into user_tags (user_id, title)
    select distinct user_id, 'important-auto'
    from important_entries
    on conflict (user_id, title) do update set title = excluded.title
    returning id, user_id
),
removed_tags as (
    delete from entry_user_tags eut
    using user_tags ut
    where eut.entry_id = any(%(ordinary_article_ids)s)
      and eut.user_tag_id = ut.id
      and ut.title = 'important-auto'
)
insert into entry_user_tags (entry_id, user_tag_id)
select e.id, ut.id
from important_entries e
join important_tags ut on ut.user_id = e.user_id
on conflict (entry_id, user_tag_id) do nothing;
