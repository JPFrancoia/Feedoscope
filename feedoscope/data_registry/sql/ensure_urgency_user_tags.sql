-- Fetch the tag IDs for the two urgency user tags (user_id=1).
-- The tags must already exist (created by the companion upsert query).
SELECT id, title
FROM user_tags
WHERE user_id = 1
AND title IN ('0-urgency', '1-urgency');
