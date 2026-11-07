create table importance_prompt (
    id bigserial primary key,
    prompt jsonb not null check (jsonb_typeof(prompt) = 'object'),
    created_at timestamp with time zone not null default now(),
    model text not null check (model ~ '^jev-[0-9]+\.[0-9]+\.[0-9]+$')
);

create function prevent_importance_prompt_update() returns trigger
language plpgsql as $$
begin
    raise exception 'Importance prompts are immutable. Insert a new version.';
end;
$$;

create trigger importance_prompt_immutable
before update on importance_prompt
for each row execute function prevent_importance_prompt_update();

create table entries_importance (
    entry_id bigint primary key references entries(id) on delete cascade,
    importance_prompt_id bigint not null references importance_prompt(id) on delete restrict,
    is_important boolean not null,
    importance_probability double precision not null
        check (importance_probability >= 0 and importance_probability <= 1),
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    inferred_at timestamp with time zone not null
);

create index entries_importance_prompt_idx on entries_importance(importance_prompt_id);
