with manual_labels as (
    select
        eut.entry_id,
        array_agg(ut.title order by ut.title) as titles
    from entry_user_tags eut
    join user_tags ut on ut.id = eut.user_tag_id
    where ut.user_id = 1
      and ut.title in ('fresh_d', 'fresh_m', 'fresh_y')
    group by eut.entry_id
), effective_labels as (
    select
        e.*,
        f.title as feed_name,
        case
            when e.status = 'read' and cardinality(ml.titles) = 1 then ml.titles[1]
            else bl.label
        end as freshness_label,
        case
            when e.status = 'read' and cardinality(ml.titles) = 1 then 'manual'
            else 'bootstrap:' || bl.source
        end as label_source
    from entries e
    join feeds f on f.id = e.feed_id
    left join manual_labels ml on ml.entry_id = e.id
    left join freshness_bootstrap_labels bl on bl.article_id = e.id
    where (
          e.status <> 'read'
          or cardinality(coalesce(ml.titles, array[]::text[])) <= 1
      )
      and (
          bl.article_id is not null
          or (e.status = 'read' and cardinality(ml.titles) = 1)
      )
)
select
    id as article_id,
    title,
    starred,
    feed_name,
    content,
    url as link,
    author,
    published_at as date_entered,
    changed_at as last_read,
    null as time_sensitivity_score,
    coalesce(tags, array[]::text[]) as tags,
    vote,
    status,
    array_position(array['fresh_d', 'fresh_m', 'fresh_y'], freshness_label) - 1
        as freshness_label,
    label_source
from effective_labels
order by published_at asc, id asc;
