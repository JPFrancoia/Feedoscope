-- Update title for an article in Miniflux
-- The title column is directly in the entries table
update entries set title=%(title)s where id = %(int_id)s;
