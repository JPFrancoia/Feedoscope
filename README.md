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

### Results

My best model so far is a Random Forest classifier + PU bagging. It achieves
the following results:

Precision for random_forest_bagging: 0.78
Recall for random_forest_bagging: 0.50
F1 score for random_forest_bagging: 0.61
ROC AUC for random_forest_bagging: 0.67
Average Precision for random_forest_bagging: 0.73
Log Loss for random_forest_bagging: 0.70

It's not perfect, but it's a start. What really matters to me is the ranking of
articles: as long as the most relevant articles are at the top, I'm happy.

Especially since the PU approach is mostly here to solve my cold start problem,
where I don't have negative examples to train a normal binary classifier.


### Sources

- [pulearn library](https://github.com/pulearn/pulearn)
- Very nice video about Positive Unabelled (PU) learning https://www.youtube.com/watch?v=uk6SlTzfbUY
