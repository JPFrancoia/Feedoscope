-- The super-important head and its automation-owned tags are removed.
-- Removing user_tags cascades to their entry_user_tags rows.
delete from user_tags where title = 'important-auto';
drop table if exists super_important_inference;
