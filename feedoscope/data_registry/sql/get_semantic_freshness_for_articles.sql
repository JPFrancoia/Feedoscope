select article_id, bucket_probabilities, expected_lifetime_days
from semantic_freshness_inference
where article_id = any(%(article_ids)s)
  and model_key = %(model_key)s;
