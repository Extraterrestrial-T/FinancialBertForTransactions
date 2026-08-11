# Experiment story

This project began as an attempt to build a small BERT-like encoder for
financial transactions by hand. The initial Indian transaction dataset was
useful for learning EDA and tokenization, but its short, sparse histories made
it a poor fit for an event-history encoder. A larger MBD archive looked more
promising, but its categorical representation could not reliably be decoded
back to its original values. The Czech dataset became the practical middle
ground: it has one million structured events, stable account histories, and
relational profile data that can become time-aware life-long state.

The first implementation focused too narrowly on a transaction MLM objective.
The architecture was then expanded into separate Event, Profile, and History
Encoders, connected so masked-field loss has a route through the whole
backbone. The data handler was changed at the same time: instead of a static
profile, it materializes a profile snapshot at each cutoff and hides life-long
events that have not yet happened.

Downstream work follows a strict sequence. First create a cutoff-safe task
table, then establish a tabular baseline, then test frozen account embeddings,
and only then decide whether an adapter is justified. This made one negative
result useful: the observed loan-trouble task has too few held-out positives
and tabular features win, so it is not a responsible fine-tuning target.

Cash-flow stress demonstrates that a frozen representation can come close to
engineered account features. The future 180-day absolute transaction-volume
proxy is the more persuasive next step because the frozen embedding has a
small edge over the Ridge baseline. The next experiments use a rank-selected,
History Encoder-only LoRA adapter and compare it with both prior baselines on
the same cached, account-disjoint task table. This is still exploratory
representation learning—not real LTV, credit scoring, or fraud detection.
