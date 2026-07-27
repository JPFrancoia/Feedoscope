alter table model_evals
    alter column metrics_accuracy drop not null,
    alter column metrics_precision drop not null,
    alter column metrics_recall drop not null,
    alter column metrics_roc_auc drop not null,
    alter column metrics_average_precision drop not null,
    alter column metrics_log_loss drop not null,
    add column metrics_rps double precision,
    add column metrics_weighted_kappa double precision,
    add column metrics_log_duration_mae double precision;
