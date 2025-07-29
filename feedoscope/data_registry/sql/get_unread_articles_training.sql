select
    e.id as article_id,
    e.title,
    ue.marked,
    f.title as feed_name,
    e.content,
    e.link,
    e.author,
    e.date_entered,
    ue.last_read,
    STRING_AGG(distinct l.caption, ', ') as labels,
    STRING_AGG(distinct t.tag_name, ', ') as tags
from
    ttrss_entries e
    join ttrss_user_entries ue on e.id = ue.ref_id
    join ttrss_feeds f on ue.feed_id = f.id
    left join ttrss_user_labels2 ul on e.id = ul.article_id
    left join ttrss_labels2 l on ul.label_id = l.id
    left join ttrss_tags t on ue.int_id = t.post_int_id
where
    ue.published = false
    and (ue.marked = false
        and ue.unread = true)
    and e.date_entered > now() - interval '1 year'
group by
    e.id,
    e.title,
    ue.marked,
    f.title,
    e.link,
    e.author,
    e.date_entered,
    ue.last_read
order by
    e.id desc
limit 2000;

