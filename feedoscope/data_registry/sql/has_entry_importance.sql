select exists(
    select 1 from entries_importance where entry_id = %(entry_id)s
) as found;
