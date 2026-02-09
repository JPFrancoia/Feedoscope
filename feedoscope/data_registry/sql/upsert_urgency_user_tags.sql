-- Ensure the two urgency user tags exist for user_id=1.
-- Idempotent: does nothing if they already exist.
INSERT INTO user_tags (user_id, title)
VALUES (1, '0-urgency'), (1, '1-urgency')
ON CONFLICT (user_id, title) DO NOTHING;
