insert into entries_importance (
    entry_id, importance_prompt_id, is_important,
    importance_probability, confidence, inferred_at
)
values (
    %(entry_id)s, %(importance_prompt_id)s, %(is_important)s,
    %(importance_probability)s, %(confidence)s, %(inferred_at)s
)
on conflict (entry_id) do nothing;
