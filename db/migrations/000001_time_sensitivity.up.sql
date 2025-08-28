create type confidence as enum (
    'low',
    'medium',
    'high'
);

create table time_sensitivity (
    article_id integer primary key references ttrss_entries(id) on delete cascade,
    score integer not null default 0,
    last_updated timestamp with time zone not null default now(),
    confidence confidence,
    explanation text
);
