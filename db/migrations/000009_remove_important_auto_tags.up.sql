-- The retired Feedoscope-owned tag is not reader data.
-- Removing user_tags cascades to their entry_user_tags rows.
delete from user_tags where title = 'important-auto';
