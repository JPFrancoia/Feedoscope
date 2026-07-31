do $$
begin
    if exists (
        select 1
        from user_tags
        where user_id = 1
          and title in (
              'fresh-lt-24h', 'fresh-1-3d', 'fresh-4-7d',
              'fresh-8-30d', 'fresh-1-6m', 'fresh-evergreen',
              'fresh-auto-lt-24h', 'fresh-auto-1-3d', 'fresh-auto-4-7d',
              'fresh-auto-8-30d', 'fresh-auto-1-6m', 'fresh-auto-evergreen'
          )
    ) then
        raise exception 'freshness tag rename blocked: fresh-* tags already exist';
    end if;
end $$;

update user_tags
set title = case title
    when 'lt-24h-freshness' then 'fresh-lt-24h'
    when '1-3d-freshness' then 'fresh-1-3d'
    when '4-7d-freshness' then 'fresh-4-7d'
    when '8-30d-freshness' then 'fresh-8-30d'
    when '1-6m-freshness' then 'fresh-1-6m'
    when 'evergreen-freshness' then 'fresh-evergreen'
    when 'lt-24h-auto-freshness' then 'fresh-auto-lt-24h'
    when '1-3d-auto-freshness' then 'fresh-auto-1-3d'
    when '4-7d-auto-freshness' then 'fresh-auto-4-7d'
    when '8-30d-auto-freshness' then 'fresh-auto-8-30d'
    when '1-6m-auto-freshness' then 'fresh-auto-1-6m'
    when 'evergreen-auto-freshness' then 'fresh-auto-evergreen'
end
where user_id = 1
  and title in (
      'lt-24h-freshness', '1-3d-freshness', '4-7d-freshness',
      '8-30d-freshness', '1-6m-freshness', 'evergreen-freshness',
      'lt-24h-auto-freshness', '1-3d-auto-freshness',
      '4-7d-auto-freshness', '8-30d-auto-freshness',
      '1-6m-auto-freshness', 'evergreen-auto-freshness'
  );
