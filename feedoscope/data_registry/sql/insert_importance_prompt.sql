insert into importance_prompt (prompt, model)
values (%(prompt)s, %(model)s)
returning id, prompt, created_at, model;
