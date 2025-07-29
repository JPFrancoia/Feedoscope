with numbered_articles as (
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
        STRING_AGG(distinct t.tag_name, ', ') as tags,
        row_number() over (order by e.id desc) as rn
    from
        ttrss_entries e
        join ttrss_user_entries ue on e.id = ue.ref_id
        join ttrss_feeds f on ue.feed_id = f.id
        left join ttrss_user_labels2 ul on e.id = ul.article_id
        left join ttrss_labels2 l on ul.label_id = l.id
        left join ttrss_tags t on ue.int_id = t.post_int_id
    where
        ue.published = false
        and (ue.marked = true
            or ue.unread = false)
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
)
select
    article_id,
    title,
    marked,
    feed_name,
    content,
    link,
    author,
    date_entered,
    last_read,
    labels,
    tags
from
    numbered_articles
where
    rn <= 100
order by
    article_id desc;

