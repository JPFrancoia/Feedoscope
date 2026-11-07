-- Restore the final table structures, not the deleted records.
create table urgency_inference (
    article_id bigint not null references entries(id) on delete cascade,
    urgency_score double precision not null check (urgency_score >= 0 and urgency_score <= 1),
    last_updated timestamp with time zone not null default now(),
    model_key text not null,
    primary key (article_id, model_key)
);

create table time_sensitivity (
    article_id bigint primary key references entries(id) on delete cascade,
    score integer not null default 0,
    last_updated timestamp with time zone not null default now(),
    confidence confidence,
    explanation text
);

create table freshness_bootstrap_labels (
    article_id bigint primary key references entries(id) on delete cascade,
    label text not null check (label in ('fresh_d', 'fresh_m', 'fresh_y')),
    source text not null,
    labeled_at timestamp with time zone not null default now()
);

create table time_sensitivity_simplified (
    article_id bigint primary key references entries(id) on delete cascade,
    score smallint not null check (score in (0, 1)),
    explanation text,
    last_updated timestamp with time zone not null default now()
);
