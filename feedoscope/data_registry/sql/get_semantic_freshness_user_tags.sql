select id, title
from user_tags
where user_id = 1
  and title in (
      'fresh-lt-24h', 'fresh-1-3d', 'fresh-4-7d',
      'fresh-8-30d', 'fresh-1-6m', 'fresh-evergreen',
      'fresh-auto-lt-24h', 'fresh-auto-1-3d',
      'fresh-auto-4-7d', 'fresh-auto-8-30d',
      'fresh-auto-1-6m', 'fresh-auto-evergreen'
  );
