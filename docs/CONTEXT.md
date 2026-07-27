# Feedoscope Context

Canonical language for Feedoscope relevance, urgency, and semantic-freshness discussions.

## Language

**Intrinsic semantic freshness**:
The length of time an article's main current or actionable claim remains useful.
_Avoid_: Urgency, importance

**Urgency**:
How quickly an article calls for attention or action.
_Avoid_: Freshness, expiry

**Active unread set**:
Unread articles currently selected for routine Feedoscope scoring, including recent articles and a rotating older sample.
_Avoid_: All articles, complete unread history

**Reviewed freshness tag**:
A freshness-horizon tag accepted through an article being read or explicitly set by the user.
_Avoid_: Human tag, manual override

**Automatic freshness tag**:
A freshness-horizon tag added by the classifier to show its prediction before review.
_Avoid_: Reviewed tag

**Bootstrap teacher label**:
An initial freshness label produced by the stronger teacher used to create the first training set.
_Avoid_: Human label, classifier prediction

## Relationships

- **Intrinsic semantic freshness** is distinct from **Urgency**.
- A **Reviewed freshness tag** takes precedence over an **Automatic freshness tag**.
- Reading an article means its freshness horizon is accepted for future training.
- A **Bootstrap teacher label** is used only when no **Reviewed freshness tag** exists.
- The **Active unread set** is the population receiving routine automatic predictions.

## Example dialogue

> **Dev:** "The article has `8-30d-auto-freshness`, but you added `1-6m-freshness`. Which label wins?"
> **Domain expert:** "The reviewed freshness tag wins. The automatic tag may remain visible so I can still see what the classifier predicted."

## Flagged ambiguities

- "Freshness" and "urgency" were initially used interchangeably; they are distinct concepts.
- "All articles" was ambiguous; routine automatic tagging targets the **Active unread set**, not the complete database.
- "Human freshness tag" originally meant personally added; resolved: a no-`auto` tag is a **Reviewed freshness tag** and may also result from accepting an automatic tag by reading the article.
