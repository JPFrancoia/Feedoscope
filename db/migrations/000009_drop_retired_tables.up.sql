-- These retired pipelines have no active readers or writers.
-- This migration deletes their stored data. Rollback restores empty tables only.
drop table urgency_inference;
drop table time_sensitivity;
drop table freshness_bootstrap_labels;
drop table time_sensitivity_simplified;
