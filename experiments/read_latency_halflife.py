"""Estimate decay half-lives from observed read latency instead of LLM labels.

For an exponential decay, the median delay between publication and read IS the
half-life. This measures the quantity `main.decay_relevance_score` needs, with
no labels and no model.

Run:
    DATABASE_URL=... uv run python -m experiments.read_latency_halflife
    uv run python -m experiments.read_latency_halflife --self-check
"""

import os
import sys

import numpy as np
import psycopg
from psycopg.rows import dict_row

MIN_READS_PER_FEED = 30

QUERY = """
with reads as (
    select
        f.title as feed_name,
        extract(epoch from (e.changed_at - e.published_at)) / 86400.0 as delay_days,
        date_trunc('second', e.changed_at) as read_second
    from entries e
    join feeds f on f.id = e.feed_id
    where e.status = 'read'
      and e.vote >= 0
      and e.changed_at > e.published_at
      and e.published_at > now() - interval '2 years'
),
-- Drop "mark all as read" bursts: they measure a click, not interest decay.
bulk as (
    select read_second from reads group by read_second having count(*) > 30
)
select feed_name, delay_days
from reads
where read_second not in (select read_second from bulk)
  and delay_days between 0.0007 and 365;
"""


def variance_explained_by_feed(feeds: np.ndarray, log_delays: np.ndarray) -> float:
    """Fraction of log-latency variance explained by the feed alone."""
    total = float(np.sum((log_delays - log_delays.mean()) ** 2))
    if total == 0.0:
        return 0.0
    within = 0.0
    for feed in np.unique(feeds):
        group = log_delays[feeds == feed]
        within += float(np.sum((group - group.mean()) ** 2))
    return 1.0 - within / total


def self_check() -> None:
    feeds = np.array(["fast"] * 50 + ["slow"] * 50)
    log_delays = np.concatenate([np.full(50, 0.1), np.full(50, 5.0)])
    assert variance_explained_by_feed(feeds, log_delays) == 1.0
    noise = np.tile([-1.0, 1.0], 50)
    assert variance_explained_by_feed(feeds, log_delays + noise) < 0.9
    assert variance_explained_by_feed(feeds, np.zeros(100)) == 0.0
    print("self-check passed")


def main() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        rows = conn.execute(QUERY).fetchall()

    feeds = np.array([row["feed_name"] for row in rows])
    delays = np.array([float(row["delay_days"]) for row in rows])
    print(f"{len(delays)} reads across {len(np.unique(feeds))} feeds\n")
    print(f"global half-life  = {np.median(delays):.2f} days")
    print(f"quartiles (days)  = {np.percentile(delays, [25, 50, 75]).round(2)}\n")

    per_feed = []
    for feed in np.unique(feeds):
        group = delays[feeds == feed]
        if len(group) >= MIN_READS_PER_FEED:
            per_feed.append((float(np.median(group)), len(group), feed))
    per_feed.sort()

    print(f"per-feed half-life (feeds with >= {MIN_READS_PER_FEED} reads):")
    for half_life, count, feed in per_feed:
        print(f"  {half_life:8.2f} d  n={count:5d}  {feed[:60]}")

    explained = variance_explained_by_feed(feeds, np.log1p(delays))
    print(f"\nvariance of log-latency explained by feed: {explained:.1%}")
    print(
        "If this is high, a per-feed lookup replaces the freshness model.\n"
        "If it is low, the text may add signal - but check the per-feed spread "
        "above is real before training anything."
    )


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
