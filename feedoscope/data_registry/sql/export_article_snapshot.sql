SELECT
    e.id AS article_id,
    e.user_id,
    e.feed_id,
    e.hash,
    e.published_at,
    e.created_at,
    e.changed_at,
    e.title,
    e.url,
    e.author,
    e.content,
    e.status::text AS status,
    e.vote,
    e.starred,
    e.saved_for_later,
    e.tags,
    ARRAY(
        SELECT t.title
        FROM entry_user_tags et
        JOIN user_tags t ON t.id = et.user_tag_id
        WHERE et.entry_id = e.id
        ORDER BY t.id
    ) AS user_tags,
    e.comments_url,
    e.reading_time,
    e.score,
    e.raw_relevance_score,
    f.title AS feed_name,
    f.site_url AS feed_site_url,
    f.description AS feed_description,
    f.category_id,
    c.title AS category_name,
    (e.status = 'read' AND e.vote >= 0) AS is_good,
    (e.vote = -1) AS is_bad,
    (e.vote >= 0 AND (e.starred OR e.vote = 1)) AS is_important
FROM entries e
LEFT JOIN feeds f ON f.id = e.feed_id
LEFT JOIN categories c ON c.id = f.category_id
ORDER BY e.id;
