delete from entry_user_tags
where user_tag_id in (
    select id
    from user_tags
    where user_id = 1 and title in ('fresh_d', 'fresh_m', 'fresh_y')
);

delete from user_tags
where user_id = 1 and title in ('fresh_d', 'fresh_m', 'fresh_y');

drop table freshness_bootstrap_labels;
