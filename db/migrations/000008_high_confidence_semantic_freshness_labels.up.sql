delete from semantic_freshness_teacher_labels
where confidence <> 'high';

alter table semantic_freshness_teacher_labels
    drop constraint semantic_freshness_teacher_labels_confidence_check,
    add constraint semantic_freshness_teacher_labels_confidence_check
        check (confidence = 'high');
