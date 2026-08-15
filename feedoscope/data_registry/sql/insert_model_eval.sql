insert into model_evals (
    eval_date,
    model,
    evaluation_model,
    training,
    eval,
    metrics
)
values (
    %(eval_date)s,
    %(model_name)s,
    %(evaluation_model)s,
    %(training)s,
    %(eval_counts)s,
    %(metrics)s
);
