-- Assign an urgency user tag (0-urgency or 1-urgency) to an entry.
--
-- This query is intentionally INSERT-only with ON CONFLICT DO NOTHING.
-- If the entry already has an urgency tag (whether auto-assigned by the LLM
-- pipeline or manually set by the user), it is NOT overwritten.
--
-- This is critical because the user may manually re-tag articles (especially
-- read ones) to correct LLM misclassifications. Those manual corrections
-- must be preserved across pipeline re-runs.
--
-- To change an article's urgency tag, do it manually via Miniflux or SQL.
INSERT INTO entry_user_tags (entry_id, user_tag_id)
SELECT %(entry_id)s, %(user_tag_id)s
WHERE NOT EXISTS (
    -- Skip if this entry already has ANY urgency tag (0-urgency or 1-urgency)
    SELECT 1 FROM entry_user_tags eut
    JOIN user_tags ut ON ut.id = eut.user_tag_id
    WHERE eut.entry_id = %(entry_id)s
    AND ut.user_id = 1
    AND ut.title IN ('0-urgency', '1-urgency')
);
