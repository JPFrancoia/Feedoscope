insert into entries_importance (
    entry_id, importance_prompt_id, is_important,
    importance_probability, confidence, inferred_at
)
values (
    %(entry_id)s, %(importance_prompt_id)s, %(is_important)s,
    %(importance_probability)s, %(confidence)s, %(inferred_at)s
)
on conflict (entry_id) do update set
    importance_prompt_id = excluded.importance_prompt_id,
    is_important = excluded.is_important,
    importance_probability = excluded.importance_probability,
    confidence = excluded.confidence,
    inferred_at = excluded.inferred_at
where excluded.importance_prompt_id > entries_importance.importance_prompt_id
    or (excluded.importance_prompt_id = entries_importance.importance_prompt_id
        and excluded.inferred_at >= entries_importance.inferred_at);
