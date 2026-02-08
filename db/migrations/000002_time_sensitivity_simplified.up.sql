create table time_sensitivity_simplified (
    article_id bigint primary key references entries(id) on delete cascade,
    score smallint not null check (score in (0, 1)),
    explanation text,
    last_updated timestamp with time zone not null default now()
);
