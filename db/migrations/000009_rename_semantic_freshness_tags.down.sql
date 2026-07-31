do $$
begin
    if exists (
        select 1
        from user_tags
        where user_id = 1
          and title in (
              'lt-24h-freshness', '1-3d-freshness', '4-7d-freshness',
              '8-30d-freshness', '1-6m-freshness', 'evergreen-freshness',
              'lt-24h-auto-freshness', '1-3d-auto-freshness',
              '4-7d-auto-freshness', '8-30d-auto-freshness',
              '1-6m-auto-freshness', 'evergreen-auto-freshness'
          )
    ) then
        raise exception 'freshness tag rollback blocked: old tags already exist';
    end if;
end $$;

update user_tags
set title = case title
    when 'fresh-lt-24h' then 'lt-24h-freshness'
    when 'fresh-1-3d' then '1-3d-freshness'
    when 'fresh-4-7d' then '4-7d-freshness'
    when 'fresh-8-30d' then '8-30d-freshness'
    when 'fresh-1-6m' then '1-6m-freshness'
    when 'fresh-evergreen' then 'evergreen-freshness'
    when 'fresh-auto-lt-24h' then 'lt-24h-auto-freshness'
    when 'fresh-auto-1-3d' then '1-3d-auto-freshness'
    when 'fresh-auto-4-7d' then '4-7d-auto-freshness'
    when 'fresh-auto-8-30d' then '8-30d-auto-freshness'
    when 'fresh-auto-1-6m' then '1-6m-auto-freshness'
    when 'fresh-auto-evergreen' then 'evergreen-auto-freshness'
end
where user_id = 1
  and title in (
      'fresh-lt-24h', 'fresh-1-3d', 'fresh-4-7d',
      'fresh-8-30d', 'fresh-1-6m', 'fresh-evergreen',
      'fresh-auto-lt-24h', 'fresh-auto-1-3d', 'fresh-auto-4-7d',
      'fresh-auto-8-30d', 'fresh-auto-1-6m', 'fresh-auto-evergreen'
  );
