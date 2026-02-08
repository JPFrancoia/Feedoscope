create table urgency_inference (
    article_id bigint primary key references entries(id) on delete cascade,
    urgency_score double precision not null check (urgency_score >= 0 and urgency_score <= 1),
    last_updated timestamp with time zone not null default now()
);
