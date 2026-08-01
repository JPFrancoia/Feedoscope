-- Downvoted unread articles are excluded from inference, so clear scores left by earlier runs.
update entries
set score = 0
where status = 'unread'
  and vote = -1
  and score <> 0;
