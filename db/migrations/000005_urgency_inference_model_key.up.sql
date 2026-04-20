alter table urgency_inference
    add column model_key text;

update urgency_inference
set model_key = 'legacy-modernbert';

alter table urgency_inference
    alter column model_key set not null;

alter table urgency_inference
    drop constraint urgency_inference_pkey;

alter table urgency_inference
    add primary key (article_id, model_key);
