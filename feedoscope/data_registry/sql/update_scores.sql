-- Update score for an article in Miniflux
-- The score column is directly in the entries table
update entries set score=%(score)s, raw_relevance_score=%(raw_score)s where id = %(int_id)s;
