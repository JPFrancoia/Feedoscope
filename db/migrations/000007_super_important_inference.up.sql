create table super_important_inference (
    article_id bigint not null references entries(id) on delete cascade,
    model_key text not null,
    super_important_score double precision not null
        check (super_important_score >= 0 and super_important_score <= 1),
    last_updated timestamp with time zone not null default now(),
    primary key (article_id, model_key)
);
