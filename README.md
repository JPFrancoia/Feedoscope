# Feedoscope

Feedoscope is a tool that ranks your RSS articles by their relevance to you.
It's meant to be used with [Tiny Tiny RSS](https://tt-rss.org/).

**It's under active development, and it's still very rough around the edges.**

## How it works

If you use ttrss like I do, you basically scroll through your list of
articles and stop to read some, based on their title. Eventually, you'll
star some articles that you found really interesting. The articles that you
deem uninteresting stay unread. However, there is only so much time in a
day to scroll through a list of articles, and you'll most likely miss some
of the interesting ones.

I would like to optimize this process by using machine learning to give a score
to each unread article, based on the articles you have read. In a nutshell,
here is how it works:

- Each read article is considered a positive example
- The ML uses these positive examples to learn what you find interesting
- A score is computed for each unread article
- You can then sort your unread articles by score, and read the most relevant ones first

Traditionally, this is a binary classification problem. This is normally solved
by collecting positive and negative examples (for example, liked and disliked
articles), training a classifier, and then using it to predict the class of
unread articles. In this scenario, you'd need something to differentiate like
and disliked articles.  
However, in my case, I don't have negative examples. I only stop to read some
articles (which are positive examples), and I don't read the rest. There could
be tons of interesting articles among the unread ones.

This is where **Positive Unlabeled (PU) learning** comes into play. Simply put,
PU learning tries to solve the problem of learning from positive examples only.


## Future development

Find a way to label negative examples. Train a normal binary classifiers once
enough negative examples are collected.


## Machine learning

### PU learning Results

My best model so far is a Random Forest classifier + PU bagging. It achieves
the following results:

```
Precision for random_forest_bagging: 0.78
Recall for random_forest_bagging: 0.50
F1 score for random_forest_bagging: 0.61
ROC AUC for random_forest_bagging: 0.67
Average Precision for random_forest_bagging: 0.73
Log Loss for random_forest_bagging: 0.70
```

It's not perfect, but it's a start. What really matters to me is the ranking of
articles: as long as the most relevant articles are at the top, I'm happy.

Especially since the PU approach is mostly here to solve my cold start problem,
where I don't have negative examples to train a normal binary classifier.

### LLM results

#### answerdotai/ModernBERT-base


I'm using this one
```
2025-08-08 16:12:26,089 -    DEBUG Collected 1945 good articles.                      [llm_learn.py, l 82 in train_model]
2025-08-08 16:12:26,090 -    DEBUG Collected 271 bad articles.                        [llm_learn.py, l 83 in train_model]
{'eval_loss': 0.15711967647075653, 'eval_average_precision': 0.9930855256520407, 'eval_roc_auc': 0.9547483380816715, 'eval_accuracy': 0.8896396396396397, 'eval_runtime': 11.9092, 'eval_samples_per_second': 37.282, 'eval_steps_per_second': 2.351, 'epoch': 1.0}
{'eval_loss': 0.18189534544944763, 'eval_average_precision': 0.9934653656433741, 'eval_roc_auc': 0.9574074074074074, 'eval_accuracy': 0.9527027027027027, 'eval_runtime': 11.7708, 'eval_samples_per_second': 37.72, 'eval_steps_per_second': 2.379, 'epoch': 2.0}
{'train_runtime': 327.9145, 'train_samples_per_second': 10.808, 'train_steps_per_second': 0.677, 'train_loss': 0.17560252627810916, 'epoch': 2.0}
2025-08-08 16:17:59,945 -     INFO Model saved to saved_models/answerdotai-ModernBERT-base_512_2_epochs_16_batch_size [llm_learn.py, l 151 in main]
2025-08-08 16:17:59,945 -    DEBUG Loading good and not good articles for validation... [llm_learn.py, l 157 in main]
2025-08-08 16:18:00,433 -    DEBUG Collected 50 good articles and 50 not good articles. [llm_learn.py, l 162 in main]
2025-08-08 16:18:03,151 -     INFO Precision: 0.78                                    [llm_learn.py, l 208 in main]
2025-08-08 16:18:03,152 -     INFO Recall: 1.00                                       [llm_learn.py, l 209 in main]
2025-08-08 16:18:03,152 -     INFO F1 score: 0.88                                     [llm_learn.py, l 210 in main]
2025-08-08 16:18:03,152 -     INFO ROC AUC: 0.97                                      [llm_learn.py, l 211 in main]
2025-08-08 16:18:03,152 -     INFO Average Precision: 0.95                            [llm_learn.py, l 212 in main]
2025-08-08 16:18:03,152 -     INFO Log Loss: 0.31                                     [llm_learn.py, l 213 in main]
```

```
2025-08-08 16:31:26,443 -    DEBUG Collected 1945 good articles.                      [llm_learn.py, l 82 in train_model]
2025-08-08 16:31:26,443 -    DEBUG Collected 271 bad articles.                        [llm_learn.py, l 83 in train_model]
{'eval_loss': 0.16846027970314026, 'eval_average_precision': 0.9899998111007333, 'eval_roc_auc': 0.9402659069325737, 'eval_accuracy': 0.9301801801801802, 'eval_runtime': 11.9033, 'eval_samples_per_second': 37.301, 'eval_steps_per_second': 2.352, 'epoch': 1.0}
{'eval_loss': 0.19872239232063293, 'eval_average_precision': 0.9899187664822419, 'eval_roc_auc': 0.9446343779677112, 'eval_accuracy': 0.9436936936936937, 'eval_runtime': 12.0825, 'eval_samples_per_second': 36.747, 'eval_steps_per_second': 2.317, 'epoch': 2.0}
{'eval_loss': 0.2569018304347992, 'eval_average_precision': 0.9891867845300613, 'eval_roc_auc': 0.943542260208927, 'eval_accuracy': 0.9391891891891891, 'eval_runtime': 12.0283, 'eval_samples_per_second': 36.913, 'eval_steps_per_second': 2.328, 'epoch': 3.0}
{'train_runtime': 499.5212, 'train_samples_per_second': 14.19, 'train_steps_per_second': 0.889, 'train_loss': 0.15623886520798141, 'epoch': 3.0}
2025-08-08 16:39:51,967 -     INFO Model saved to saved_models/answerdotai-ModernBERT-base_512_4_epochs_16_batch_size [llm_learn.py, l 151 in main]
2025-08-08 16:39:52,432 -    DEBUG Collected 50 good articles and 50 not good articles. [llm_learn.py, l 162 in main]
2025-08-08 16:39:55,371 -     INFO Precision: 0.81                                    [llm_learn.py, l 208 in main]
2025-08-08 16:39:55,371 -     INFO Recall: 0.96                                       [llm_learn.py, l 209 in main]
2025-08-08 16:39:55,371 -     INFO F1 score: 0.88                                     [llm_learn.py, l 210 in main]
2025-08-08 16:39:55,371 -     INFO ROC AUC: 0.96                                      [llm_learn.py, l 211 in main]
2025-08-08 16:39:55,372 -     INFO Average Precision: 0.95                            [llm_learn.py, l 212 in main]
2025-08-08 16:39:55,372 -     INFO Log Loss: 0.32                                     [llm_learn.py, l 213 in main]
```

This one is a strong contender, since I'm aiming for the highest average
precision. But the numbers are similar to the other parameters, and it might be
because my validation set is too small.
```
2025-08-08 16:52:37,621 -    DEBUG Collected 1945 good articles.                      [llm_learn.py, l 83 in train_model]
2025-08-08 16:52:37,622 -    DEBUG Collected 271 bad articles.                        [llm_learn.py, l 84 in train_model]
{'eval_loss': 0.17686720192432404, 'eval_average_precision': 0.9910157779775184, 'eval_roc_auc': 0.950997150997151, 'eval_accuracy': 0.9391891891891891, 'eval_runtime': 22.5694, 'eval_samples_per_second': 19.673, 'eval_steps_per_second': 2.481, 'epoch': 1.0}
{'eval_loss': 0.2075183093547821, 'eval_average_precision': 0.9929550838635847, 'eval_roc_auc': 0.9580246913580247, 'eval_accuracy': 0.9459459459459459, 'eval_runtime': 22.3909, 'eval_samples_per_second': 19.83, 'eval_steps_per_second': 2.501, 'epoch': 2.0}
{'train_runtime': 646.0572, 'train_samples_per_second': 5.486, 'train_steps_per_second': 0.687, 'train_loss': 0.2058671659177488, 'epoch': 2.0}
2025-08-08 17:03:29,710 -     INFO Model saved to saved_models/answerdotai-ModernBERT-base_1024_2_epochs_8_batch_size [llm_learn.py, l 152 in main]
2025-08-08 17:03:30,169 -    DEBUG Collected 50 good articles and 50 not good articles. [llm_learn.py, l 163 in main]
2025-08-08 17:03:37,035 -     INFO Precision: 0.79                                    [llm_learn.py, l 209 in main]
2025-08-08 17:03:37,035 -     INFO Recall: 1.00                                       [llm_learn.py, l 210 in main]
2025-08-08 17:03:37,035 -     INFO F1 score: 0.88                                     [llm_learn.py, l 211 in main]
2025-08-08 17:03:37,035 -     INFO ROC AUC: 0.98                                      [llm_learn.py, l 212 in main]
2025-08-08 17:03:37,035 -     INFO Average Precision: 0.98                            [llm_learn.py, l 213 in main]
2025-08-08 17:03:37,035 -     INFO Log Loss: 0.30                                     [llm_learn.py, l 214 in main]


#### answerdotai/ModernBERT-large


### GPU vs CPU

for 250 + 1900 articles (roughly), it takes around 60 min to train
answerdotai/ModernBERT-base with max length 512 and

```python
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
```

With a RTX 3060, same conditions, around 6 min to train.

### Sources

- [pulearn library](https://github.com/pulearn/pulearn)
- Very nice video about Positive Unabelled (PU) learning https://www.youtube.com/watch?v=uk6SlTzfbUY
