# FIRM-4B v2.3 depth-and-precision experiment

## Motivation

FIRM-4B v2.2 fixed the major thinking-loop / no-final-answer failure, but manual use
showed a different weakness: user-visible answers are often too terse and several
long-tail packaging/material facts remained wrong.

Observed v2.2 failures included:
- TO-66 incorrectly described as a plastic/epoxy package,
- TO headers described generically as brass,
- KRS-5 misidentified as a Ge-As semiconductor,
- HgCdTe incorrectly called indirect-gap,
- comparisons such as HgCdTe vs InSb collapsing to taxonomy-only answers.

The v2.2 curriculum had a median target answer length of only about 29 words and
95% of targets were below about 42 words. v2.3 therefore targets adaptive
completeness rather than generic verbosity.

## Curriculum

v2.3 inherits the complete v2.2 curriculum and adds:
- 384 complete-answer examples with 80-164 word targets,
- explicit TO/Kovar/glass-seal packaging coverage,
- KRS-5 optical-material coverage,
- direct-gap HgCdTe correction,
- multi-axis HgCdTe vs InSb comparisons,
- expanded detector/noise/process explanations,
- 64 precision-correction examples,
- 64 short-reasoning / complete-final-answer examples,
- an adaptive-depth system policy.

Current candidate:
- 5,650 train
- 491 validation
- 6,141 total
- overall p95 answer length: 115 words
- depth-example median: 118 words
- longest rendered training row: 509 tokens (1024-token limit)

Exact user regression prompts are excluded from training and stored only in the
ignored private-eval area. Public held-out evaluation is
`evals/firm_v23_depth_gauntlet_v1.jsonl`.

## Training

Start again from the exact pinned Qwen3.5-4B base:
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

- BF16 LoRA
- rank 16 / alpha 32
- LR 1.5e-5
- effective batch 4
- max sequence 1024
- 1,413 optimizer steps (~one effective pass)
- mixed non-thinking and short-thinking SFT
- final-answer completeness is trained separately from reasoning length

## Release gates

v2.3 must:
1. preserve v2.2's short, terminating thinking,
2. materially increase answer completeness on comparison/engineering prompts,
3. correct TO/Kovar/KRS-5/direct-gap failures,
4. avoid hallucinated package/process specifics,
5. preserve quantitative and scope behavior,
6. remain practical as Q4_K_M in Ollama,
7. outperform v2.2 on the new depth gauntlet and private regressions before promotion.
