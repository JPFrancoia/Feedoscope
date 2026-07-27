select id, title
from user_tags
where user_id = 1
  and title in (
      'lt-24h-freshness', '1-3d-freshness', '4-7d-freshness',
      '8-30d-freshness', '1-6m-freshness', 'evergreen-freshness',
      'lt-24h-auto-freshness', '1-3d-auto-freshness',
      '4-7d-auto-freshness', '8-30d-auto-freshness',
      '1-6m-auto-freshness', 'evergreen-auto-freshness'
  );
