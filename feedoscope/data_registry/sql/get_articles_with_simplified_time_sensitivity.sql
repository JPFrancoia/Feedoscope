-- Get all articles that have a simplified time sensitivity score.
-- Used for training the distilled urgency ModernBERT model.
-- Returns article data joined with the binary urgency label.
-- Manual user tags (0-urgency, 1-urgency) override the LLM-assigned score
-- from time_sensitivity_simplified, allowing human corrections.
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
    COALESCE(e.tags, array[]::text[]) as tags,
    e.vote,
    COALESCE(
        CASE ut.title
            WHEN '0-urgency' THEN 0
            WHEN '1-urgency' THEN 1
        END,
        tss.score
    ) as urgency_label
from
    entries e
    join feeds f on e.feed_id = f.id
    join time_sensitivity_simplified tss on tss.article_id = e.id
    left join entry_user_tags eut on eut.entry_id = e.id
    left join user_tags ut on ut.id = eut.user_tag_id
        and ut.user_id = 1
        and ut.title in ('0-urgency', '1-urgency')
order by
    e.id asc;
