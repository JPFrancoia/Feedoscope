# Why urgency probability is not semantic lifetime

## Short answer

The existing binary classifier may be a reasonable **urgent/not-urgent classifier**. The experiment showed that its probability is a poor estimate of **how long an article stays useful**.

Calibration might contribute, but it is not the main problem. The main problem is that the classifier was never trained on duration.

## Why the binary urgency score is insufficient

A binary model learns:

> How likely is this article to receive the `urgent` label?

Even if its output is perfectly calibrated, `0.8` means:

> Among similar articles scored 0.8, about 80% are labelled urgent.

It does not mean that the article expires in 20 days, has used 80% of its useful life, or has an 80% chance of expiring today. The model never received that information during training.

Different lifetimes can also have the same urgency:

| Article | Urgency | Likely lifetime |
|---|---|---|
| Live election results | High | Hours |
| Critical software vulnerability | High | Weeks or months |
| Tax deadline in three months | Low today | Three months |
| Python tutorial | Low | Evergreen |

The binary classifier may score both of the first two near `1`, although their lifetimes are very different. Likewise, "not urgent" is broader than "evergreen": an article can be non-urgent but still expire next week.

One urgency number therefore collapses several different concepts:

- how quickly action is needed;
- how soon the information becomes outdated;
- how important it is;
- whether it is evergreen.

## What the new method does differently

The new method still uses simple logistic regression. The important change is what the models are taught to predict.

For every article, it asks five concrete questions:

1. Will it remain useful beyond 24 hours?
2. Beyond 3 days?
3. Beyond 7 days?
4. Beyond 30 days?
5. Beyond 6 months?

For an article expected to last 8–30 days, the desired answers are:

```text
Beyond 24 hours?  Yes
Beyond 3 days?    Yes
Beyond 7 days?    Yes
Beyond 30 days?   No
Beyond 6 months?  No
```

Those five probabilities are converted into probabilities for six lifetime buckets:

1. less than 24 hours;
2. 1–3 days;
3. 4–7 days;
4. 8–30 days;
5. 1–6 months;
6. evergreen (more than 6 months).

This is better because the training target now matches the question we want answered.

## Why five classifiers help

Different clues matter at different boundaries:

- "live," "breaking," and developing events help distinguish hours from days;
- deadlines and scheduled events help distinguish days from weeks;
- tutorials and reference material help distinguish months from evergreen.

A single urgent/not-urgent boundary must compress all those distinctions into one line. The new method can learn a different semantic boundary for each lifetime threshold.

It also respects the natural order:

```text
P(useful beyond 24 hours)
    >= P(useful beyond 3 days)
    >= P(useful beyond 7 days)
    >= P(useful beyond 30 days)
    >= P(useful beyond 6 months)
```

The selected experiment sorts the five values so they cannot contradict this rule, then subtracts adjacent values to obtain the six bucket probabilities.

## Is calibration still relevant?

Yes, but calibration answers a narrower question.

A binary urgency model is calibrated if, among articles predicted as 70% urgent, roughly 70% receive the urgent label. Calibration can be damaged by class weighting, class imbalance, changes in article distribution, regularization, or noisy labels.

Platt scaling or isotonic calibration may improve that binary probability. Calibration cannot, however, recover information absent from the labels. A perfectly calibrated urgent/not-urgent probability still cannot distinguish "urgent for six hours" from "urgent for six weeks." Duration labels are needed for that.

## What the benchmark showed

The benchmark did not prove that the production classifier is bad at detecting urgency. It showed that using urgency probability as a shortcut for lifetime performs poorly:

- urgency probability mapped to lifetime: RPS `0.21905157`;
- model trained directly on ordered lifetime: RPS `0.08682275`.

Lower RPS is better. The new model reduced RPS by about 60% on the frozen temporal benchmark.

The practical conclusion is:

> Keep urgency probability for urgency decisions. Use a separately trained ordered-lifetime model for freshness or expiry decisions.

The result is based on teacher-generated labels and a small fixed test set. It should be confirmed once on a new, untouched, preferably human-reviewed temporal holdout before production use.
