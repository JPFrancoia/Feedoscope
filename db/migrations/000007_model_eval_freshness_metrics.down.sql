delete from model_evals where model = 'Freshness';

alter table model_evals
    drop column metrics_log_duration_mae,
    drop column metrics_weighted_kappa,
    drop column metrics_rps;

alter table model_evals
    alter column metrics_accuracy set not null,
    alter column metrics_precision set not null,
    alter column metrics_recall set not null,
    alter column metrics_roc_auc set not null,
    alter column metrics_average_precision set not null,
    alter column metrics_log_loss set not null;
