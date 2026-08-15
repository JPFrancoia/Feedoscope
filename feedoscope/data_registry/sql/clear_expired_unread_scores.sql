-- Clear derived scores that current-formula inference can only store as zero.
update entries
set score = 0
where status = 'unread'
  and vote != -1
  and starred = false
  and score <> 0
  and published_at <= now() - interval '1 day' * %(score_horizon_days)s;
