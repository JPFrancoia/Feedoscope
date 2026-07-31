with reviewed_labels as (
    select
        eut.entry_id,
        array_agg(ut.title order by ut.title) as titles
    from entry_user_tags eut
    join user_tags ut on ut.id = eut.user_tag_id
    where ut.user_id = 1
      and ut.title in (
          'lt-24h-freshness', '1-3d-freshness', '4-7d-freshness',
          '8-30d-freshness', '1-6m-freshness', 'evergreen-freshness'
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
                'lt-24h-freshness', '1-3d-freshness', '4-7d-freshness',
                '8-30d-freshness', '1-6m-freshness', 'evergreen-freshness'
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
