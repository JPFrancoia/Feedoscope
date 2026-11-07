select id, prompt, created_at, model
from importance_prompt
order by id desc
limit 1;
