select
    count(*) as articles,
    count(*) filter (where status = 'read' and vote >= 0) as good,
    count(*) filter (where vote = -1) as bad,
    count(*) filter (where vote >= 0 and (starred or vote = 1)) as important,
    count(*) filter (
        where status = 'read' and vote >= 0 and (starred or vote = 1)
    ) as important_read,
    coalesce(sum(id), 0)::text as article_id_sum,
    current_setting('transaction_read_only') as read_only,
    current_setting('default_transaction_read_only') as default_read_only,
    current_setting('transaction_isolation') as isolation,
    current_timestamp as snapshot_time,
    pg_current_snapshot()::text as snapshot
from entries;
