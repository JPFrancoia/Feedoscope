insert into user_tags (user_id, title)
values
    (1, 'lt-24h-freshness'),
    (1, '1-3d-freshness'),
    (1, '4-7d-freshness'),
    (1, '8-30d-freshness'),
    (1, '1-6m-freshness'),
    (1, 'evergreen-freshness'),
    (1, 'lt-24h-auto-freshness'),
    (1, '1-3d-auto-freshness'),
    (1, '4-7d-auto-freshness'),
    (1, '8-30d-auto-freshness'),
    (1, '1-6m-auto-freshness'),
    (1, 'evergreen-auto-freshness')
on conflict (user_id, title) do nothing;
