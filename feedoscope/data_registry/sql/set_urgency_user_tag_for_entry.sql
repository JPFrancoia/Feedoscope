-- Remove any existing urgency tags for this entry, then assign the new one.
WITH removed AS (
    DELETE FROM entry_user_tags
    WHERE entry_id = %(entry_id)s
    AND user_tag_id IN (
        SELECT id FROM user_tags
        WHERE user_id = 1 AND title IN ('0-urgency', '1-urgency')
    )
)
INSERT INTO entry_user_tags (entry_id, user_tag_id)
VALUES (%(entry_id)s, %(user_tag_id)s)
ON CONFLICT (entry_id, user_tag_id) DO NOTHING;
