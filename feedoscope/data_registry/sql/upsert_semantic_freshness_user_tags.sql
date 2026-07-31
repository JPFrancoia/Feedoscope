insert into user_tags (user_id, title)
values
    (1, 'fresh-lt-24h'),
    (1, 'fresh-1-3d'),
    (1, 'fresh-4-7d'),
    (1, 'fresh-8-30d'),
    (1, 'fresh-1-6m'),
    (1, 'fresh-evergreen'),
    (1, 'fresh-auto-lt-24h'),
    (1, 'fresh-auto-1-3d'),
    (1, 'fresh-auto-4-7d'),
    (1, 'fresh-auto-8-30d'),
    (1, 'fresh-auto-1-6m'),
    (1, 'fresh-auto-evergreen')
on conflict (user_id, title) do nothing;
