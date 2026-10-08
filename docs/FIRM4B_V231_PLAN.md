# FIRM-4B v2.3.1 natural-prompt calibration patch

## Why v2.3 was not promoted

v2.3 successfully increased answer length relative to v2.2, but it failed the
post-quantization release gate.

Frozen 24-item v2.3 depth/precision gauntlet:
- v2.2, thinking on: 0/24 passes; mean final answer ~20 words
- v2.3, thinking on: 1/24 passes; mean final answer ~54 words
- v2.3, thinking off: 3/24 passes; mean final answer ~55 words

Observed private regressions included:
- KRS-5 still misidentified after quantization,
- HgCdTe once mislabeled as III-VI,
- HgCdTe vs InSb remained too shallow,
- follow-ups requesting more depth often repeated the previous answer,
- thinking mode was generally worse than non-thinking mode on answer depth.

The v2.3 curriculum audit found that most long-answer rows were explicitly cued
by phrases such as "give a complete technical explanation." The model therefore
learned conditional verbosity rather than default technical completeness.

## v2.3.1 intervention

v2.3.1 is a focused calibration retrain from the untouched pinned
Qwen/Qwen3.5-4B base, not an adapter stack.

Main changes:
- natural ordinary-user prompts map directly to complete specialist answers,
- critical TO/Kovar/KRS-5/HgCdTe/InSb facts are oversampled with paraphrases,
- thinking examples pair short reasoning with complete final answers,
- multi-turn rows explicitly teach expansion after follow-ups,
- the trainer masks the full prior conversation and supervises only the final
  assistant completion,
- redundant short-form canonical/quantitative variants are downsampled while
  preserving breadth.

Final candidate corpus:
- 5,489 train
- 477 validation
- 1,541 focused calibration rows (25.8% of total)
- 1,077 natural-depth rows
- 240 natural thinking-depth rows
- 224 multi-turn expansion rows
- 1,570 canonical fact rows = 10 variants for each of 157 facts
- 1,599 quantitative rows with all quantitative families retained
- answer p75 = 114 words
- calibration median = 118 words
- longest real Qwen-tokenized sequence = 567 tokens under a 1024-token limit

## Training

Pinned base revision:
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

Configuration:
- BF16 rank-16 LoRA
- alpha 32
- learning rate 1.5e-5
- effective batch 4
- max sequence 1024
- 1,373 optimizer steps (~one effective pass)
- eval every 275 steps
- save every 125 steps
- one NVIDIA L4
- same hard-delete / no-service-account cloud envelope as v2.3

## Evaluation

v2.3.1 must beat both:
1. the original 24-item v2.3 depth gauntlet, and
2. the newly frozen 30-item shadow gauntlet
   `evals/firm_v231_shadow_gauntlet_v1.jsonl`.

Private literal user regressions remain uncommitted and excluded from train/valid.

Promotion requires:
- no KRS-5 / TO / HgCdTe taxonomy regressions,
- materially higher depth-gate pass rate in both thinking and non-thinking modes,
- no blank-answer / reasoning-loop regression,
- complete HgCdTe-vs-InSb comparisons from ordinary short prompts,
- follow-up expansion that adds new technical dimensions,
- preserved quantitative, process, diagnostic, scope and Q4_K_M behavior.
