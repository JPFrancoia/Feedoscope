insert into model_evals (
    eval_date,
    model,
    training,
    eval,
    metrics_accuracy,
    metrics_precision,
    metrics_recall,
    metrics_f1,
    metrics_roc_auc,
    metrics_average_precision,
    metrics_log_loss
)
values (
    %(eval_date)s,
    %(model_name)s,
    %(training)s,
    %(eval_counts)s,
    %(metrics_accuracy)s,
    %(metrics_precision)s,
    %(metrics_recall)s,
    %(metrics_f1)s,
    %(metrics_roc_auc)s,
    %(metrics_average_precision)s,
    %(metrics_log_loss)s
);
