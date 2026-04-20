with ranked_rows as (
    select
        ctid,
        row_number() over (
            partition by article_id
            order by last_updated desc, model_key desc
        ) as row_rank
    from urgency_inference
)
delete from urgency_inference ui
using ranked_rows rr
where ui.ctid = rr.ctid
  and rr.row_rank > 1;

alter table urgency_inference
    drop constraint urgency_inference_pkey;

alter table urgency_inference
    add primary key (article_id);

alter table urgency_inference
    drop column model_key;
