create table semantic_freshness_teacher_labels (
    article_id bigint primary key references entries(id) on delete cascade,
    horizon smallint not null check (horizon between 0 and 5),
    confidence confidence not null check (confidence in ('medium', 'high')),
    source text not null,
    labeled_at timestamp with time zone not null default now()
);

create table semantic_freshness_inference (
    article_id bigint not null references entries(id) on delete cascade,
    model_key text not null,
    bucket_probabilities double precision[] not null
        check (cardinality(bucket_probabilities) = 6),
    expected_lifetime_days double precision not null
        check (expected_lifetime_days >= 0),
    last_updated timestamp with time zone not null default now(),
    primary key (article_id, model_key)
);
