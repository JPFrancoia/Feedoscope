do $$
begin
    if exists (
        select 1
        from user_tags
        where user_id = 1 and title in ('fresh_d', 'fresh_m', 'fresh_y')
    ) then
        raise exception 'freshness migration blocked: fresh_d/fresh_m/fresh_y tags already exist';
    end if;
end $$;

create table freshness_bootstrap_labels (
    article_id bigint primary key references entries(id) on delete cascade,
    label text not null check (label in ('fresh_d', 'fresh_m', 'fresh_y')),
    source text not null,
    labeled_at timestamp with time zone not null default now()
);

insert into user_tags (user_id, title)
values (1, 'fresh_d'), (1, 'fresh_m'), (1, 'fresh_y');
