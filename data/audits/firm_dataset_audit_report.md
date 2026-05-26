# FIRM Dataset Audit

Source file: `FIRM Dataset Large - FIRM_Dataset_Sheet1_updated.csv`

## Executive assessment

The dataset is a strong seed for FIRM because it already contains 2,532 clean input/output rows, zero missing fields, no exact duplicate input/output pairs, and many technical calculation examples.

The main limitation is depth. The mean assistant answer length is only 23.4 words, while the strongest examples near the end of the file are multi-step derivations. For the next FIRM version, this dataset should be treated as a domain-anchoring and formula-behavior seed, then expanded with longer derivations, empirical workflows, measurement diagnostics, and HgCdTe-specific theory.

## Schema

Columns:

- `input`
- `output`
- `topic`
- `subtopic`
- `tags`
- `difficulty`
- `format`

The uploaded CSV has an extra blank first row and a header row stored as data. The generated processing scripts should normalize that issue.

## Core statistics

| Metric | Value |
|---|---:|
| Usable rows | 2,532 |
| Missing required cells | 0 |
| Exact duplicate inputs | 0 |
| Exact duplicate outputs | 114 |
| Exact duplicate input/output pairs | 0 |
| Mean input length | 16.4 words |
| Mean output length | 23.4 words |
| 95th percentile output length | 32.0 words |
| Maximum output length | 302 words |
| Outputs with equation-like notation | 1,941 |
| Outputs with explicit technical units | 1,847 |

## Topic distribution

| topic | count | percent |
|---|---:|---:|
| Physics | 2051 | 81.00 |
| Guardrails | 96 | 3.79 |
| Devices | 66 | 2.61 |
| Contextualization | 45 | 1.78 |
| Identity | 45 | 1.78 |
| Fabrication | 40 | 1.58 |
| Testing | 28 | 1.11 |
| Noise | 24 | 0.95 |
| Mathematics | 19 | 0.75 |
| Radiometry | 19 | 0.75 |
| Defects | 16 | 0.63 |
| Reliability | 14 | 0.55 |
| Surface | 14 | 0.55 |
| Industry | 13 | 0.51 |
| Engineering | 13 | 0.51 |
| Optics | 9 | 0.36 |
| Process | 7 | 0.28 |
| Operation | 4 | 0.16 |
| Thermal | 4 | 0.16 |
| Packaging | 3 | 0.12 |
| Modeling | 2 | 0.08 |

## Difficulty distribution

| difficulty | count |
|---|---:|
| Advanced | 1288 |
| Intermediate | 1005 |
| Beginner | 239 |

## Format distribution

| format | count |
|---|---:|
| QA | 2032 |
| Conceptual | 146 |
| Definition | 139 |
| Derivation | 70 |
| Equation | 41 |
| Code | 41 |
| List | 20 |
| Instruction-Response | 15 |
| Table | 10 |
| Comparison | 9 |
| Procedure | 7 |
| Guardrail | 1 |
| Reframing | 1 |

## Keyword coverage

| keyword_area | count | percent |
|---|---:|---:|
| HgCdTe/MCT | 123 | 4.86 |
| InSb | 6 | 0.24 |
| InAsSb | 0 | 0.00 |
| T2SL | 0 | 0.00 |
| QWIP | 2 | 0.08 |
| Photoconductor | 57 | 2.25 |
| Photodiode | 501 | 19.79 |
| Responsivity | 496 | 19.59 |
| Detectivity/D* | 171 | 6.75 |
| Noise | 823 | 32.50 |
| Johnson noise | 134 | 5.29 |
| 1/f noise | 35 | 1.38 |
| G-R noise | 166 | 6.56 |
| Shot noise | 274 | 10.82 |
| Blackbody | 25 | 0.99 |
| Lock-in/MFLI | 13 | 0.51 |
| Chopper | 9 | 0.36 |
| Cryogenic/LN2 | 26 | 1.03 |
| LPE | 16 | 0.63 |
| MOCVD | 3 | 0.12 |
| MBE | 0 | 0.00 |
| Anneal | 16 | 0.63 |
| Passivation | 37 | 1.46 |
| Readout/ROIC | 23 | 0.91 |
| FPA | 19 | 0.75 |
| FEniCSx/Digital twin | 14 | 0.55 |
| Radiometry | 156 | 6.16 |
| Bandgap/cutoff | 381 | 15.05 |

## Major finding: overrepresented formula templates

| count | example_input |
|---:|---|
| 576 | For a dopant activation energy of 0.063 eV at 155.2 K, estimate the dopant ionization fraction. |
| 180 | Given noise current 1.813e-14 A and responsivity 0.578 A/W, compute the noise equivalent power (NEP). |
| 150 | A detector with area 1.370e-08 m^2 and bandwidth 85620.2 Hz has NEP 1.069e-10 W. Compute the specific detectivity (D*). |
| 140 | Compute the shot-noise-limited SNR for a photodiode with responsivity 0.834 A/W, optical power 4.566e-03 W, and bandwidth 785.7 Hz. |
| 120 | Calculate the shot noise current for a photodiode with average current 7.376e-08 A in a bandwidth 112633.8 Hz. |
| 100 | A photodiode with quantum efficiency 0.499 operates at 425.3 nm. If it receives 4.659 mW of optical power, what is the photocurrent? |
| 100 | Compute the spectral radiance at 793.3 K and wavelength 1902.1 nm using Planck's law. |
| 100 | Calculate the Johnson noise voltage for a resistor of 2.0e+04 Ohm at 328.5 K in a bandwidth 288684.5 Hz. |
| 98 | For a dopant activation energy of 0.048 eV at 357.3 K, estimate the fraction of dopants ionized. |
| 80 | Determine the built-in potential of a silicon p-n junction with N_A = 3.58e+15 cm^-3 and N_D = 2.71e+15 cm^-3 at 300 K. |

These examples are useful, but they are overrepresented. If trained naively, FIRM may overproduce short formula-substitution answers instead of doing detailed photodetector reasoning.

## Major finding: current HgCdTe depth is too low

HgCdTe/MCT-related rows: approximately 123 rows.

That is enough to anchor the identity, but not enough for a strong HgCdTe expert model. The target should be at least:

- 500+ HgCdTe physics/theory examples
- 300+ HgCdTe fabrication/process examples
- 300+ HgCdTe measurement/troubleshooting examples
- 200+ HgCdTe empirical data-analysis examples
- 200+ HgCdTe derivation examples

## Recommended generated files

- `firm_v0_full_sft_messages.jsonl`: full normalized dataset converted to chat-message format.
- `firm_v1_balanced_sft_messages.jsonl`: deterministic downsampled training candidate that reduces repeated templates.
- `firm_v1_balanced_train.jsonl`, `firm_v1_balanced_valid.jsonl`, `firm_v1_balanced_test.jsonl`: split version of the balanced dataset.
- `firm_dataset_flags_review.csv`: review sheet with quality flags and metadata.
- `top_template_families.csv`: high-frequency template audit.
- `keyword_coverage.csv`: domain coverage audit.
- `topic_distribution.csv`: topic distribution.

## Recommended next dataset expansion themes

1. HgCdTe bandgap/composition/cutoff derivations.
2. Intrinsic carrier concentration and dark-current scaling.
3. Photoconductor gain, lifetime, transit time, and voltage responsivity.
4. Johnson, shot, 1/f, and generation-recombination noise derivations.
5. Lock-in measurement workflows with MFLI-style data.
6. Blackbody radiometry from source temperature to pixel power.
7. Cryogenic package parasitics: capacitance, leakage, thermal conduction, microphonics.
8. LPE/MOCVD/MBE HgCdTe process diagnostics.
9. Empirical curve interpretation: IV, noise PSD, spectral response, frequency response.
10. Guardrail examples that redirect generic physics toward infrared detector relevance.

## Training recommendation

Use this dataset in two stages:

1. Train a small sanity model on `firm_v1_balanced_train.jsonl` to verify formatting and behavior.
2. Expand with 1,000-3,000 deeper examples before training the serious 3B-4B local model.

The full dataset is useful, but the balanced version is safer for the first high-quality FIRM rebuild.
