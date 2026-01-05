create type confidence as enum (
    'low',
    'medium',
    'high'
);

create table time_sensitivity (
    article_id bigint primary key references entries(id) on delete cascade,
    score integer not null default 0,
    last_updated timestamp with time zone not null default now(),
    confidence confidence,
    explanation text
);
