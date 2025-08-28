select
    e.id as article_id,
    e.title,
    ue.marked,
    ue.score,
    f.title as feed_name,
    e.content,
    e.link,
    e.author,
    e.date_entered,
    ue.last_read,
    coalesce(array_agg(distinct l.caption) filter (where l.caption is not null), array[]::text[]) as labels,
    coalesce(array_agg(distinct t.tag_name) filter (where t.tag_name is not null), array[]::text[]) as tags
from
    ttrss_entries e
    join ttrss_user_entries ue on e.id = ue.ref_id
    join ttrss_feeds f on ue.feed_id = f.id
    left join ttrss_user_labels2 ul on e.id = ul.article_id
    left join ttrss_labels2 l on ul.label_id = l.id
    left join ttrss_tags t on ue.int_id = t.post_int_id
where
    e.date_entered >= now() - interval '1 day' * %(number_of_days)s
    and not exists (
        select 1 from time_sensitivity ts
        where ts.article_id = e.id
    )
group by
    e.id,
    e.title,
    ue.marked,
    ue.score,
    f.title,
    e.content,
    e.link,
    e.author,
    e.date_entered,
    ue.last_read
order by
    e.id asc;
