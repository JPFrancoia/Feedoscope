with reviewed_labels as (
    select
        eut.entry_id,
        array_agg(ut.title order by ut.title) as titles
    from entry_user_tags eut
    join user_tags ut on ut.id = eut.user_tag_id
    where ut.user_id = 1
      and ut.title in (
          'fresh-lt-24h', 'fresh-1-3d', 'fresh-4-7d',
          'fresh-8-30d', 'fresh-1-6m', 'fresh-evergreen'
      )
    group by eut.entry_id
)
select
    e.id as article_id,
    e.title,
    e.starred,
    f.title as feed_name,
    e.content,
    e.url as link,
    e.author,
    e.published_at as date_entered,
    e.changed_at as last_read,
    null as time_sensitivity_score,
    coalesce(e.tags, array[]::text[]) as tags,
    e.vote,
    e.status,
    case
        when cardinality(rl.titles) = 1 then array_position(
            array[
                'fresh-lt-24h', 'fresh-1-3d', 'fresh-4-7d',
                'fresh-8-30d', 'fresh-1-6m', 'fresh-evergreen'
            ],
            rl.titles[1]
        ) - 1
        else tl.horizon
    end as freshness_label,
    case when cardinality(rl.titles) = 1 then 'reviewed' else 'teacher' end as label_source,
    case when cardinality(rl.titles) = 1 then 'high' else tl.confidence::text end as label_confidence
from entries e
join feeds f on f.id = e.feed_id
left join reviewed_labels rl on rl.entry_id = e.id
left join semantic_freshness_teacher_labels tl on tl.article_id = e.id
where e.status = 'read'
  and cardinality(coalesce(rl.titles, array[]::text[])) <= 1
  and (
      cardinality(rl.titles) = 1
      or (rl.titles is null and tl.confidence = 'high')
  )
order by e.published_at asc, e.id asc;
