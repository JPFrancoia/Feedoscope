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


The full blog articles is here: []()
