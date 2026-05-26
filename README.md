# FIRM: Focused Infrared Research Model

FIRM is a local, domain-specialized language model project for infrared photodetectors, with emphasis on HgCdTe/MCT, detector physics, noise, radiometry, cryogenic testing, lock-in measurements, and empirical troubleshooting.

## Immediate objective

Build FIRM slowly and reproducibly from the existing dataset into a compact local model that can expand theoretical, mathematical, and empirical details related to infrared photodetectors.

## Target behavior

FIRM should answer technical questions using:

1. assumptions,
2. governing equations,
3. defined variables and units,
4. physical interpretation,
5. measurement consequences,
6. empirical caveats and failure modes.

## Repository layout

```text
FIRM/
├── configs/                 # Training configuration files
├── data/
│   ├── raw/                 # Original uploaded datasets
│   ├── processed/           # SFT/DPO-ready JSONL files
│   └── audits/              # Dataset audit outputs
├── docs/                    # Dataset/model notes
├── evals/                   # Fixed evaluation prompts
├── scripts/                 # Audit, conversion, training, evaluation scripts
└── README.md
```

## Current dataset audit snapshot

The first attached CSV audit found:

| Metric | Value |
|---|---:|
| Usable rows | 2,532 |
| Required fields missing | 0 |
| Exact duplicate input/output pairs | 0 |
| Exact duplicate inputs | 0 |
| Duplicate outputs | 114 |
| Mean input length | 16.4 words |
| Mean output length | 23.4 words |
| 95th percentile output length | 32 words |
| Max output length | 302 words |

Interpretation: the dataset is a strong compact domain anchor, but it is short-answer-heavy. The next improvement step is to add derivation-style and empirical-analysis examples.

## Recommended first training target

```text
Base model: Qwen/Qwen3-4B or Qwen/Qwen3-1.7B for faster iteration
Training method: QLoRA SFT
Initial sequence length: 2048
LoRA rank: 16 or 32
Validation split: 5 percent
Test split: 5 percent
Local deployment: GGUF Q5_K_M and Q8_0
```

## High-priority expansion areas

1. HgCdTe bandgap, composition, cutoff wavelength, and temperature dependence.
2. Intrinsic carrier concentration and dark-current scaling.
3. Photoconductor gain, lifetime, transit time, and frequency roll-off.
4. Johnson, shot, 1/f, and generation-recombination noise.
5. Blackbody radiometry and optical power coupling.
6. Lock-in amplifier measurements, MFLI workflows, and chopped-source testing.
7. LPE, MOCVD, annealing, passivation, and cryogenic packaging.
8. Empirical troubleshooting from IV curves, spectra, noise spectra, and frequency response.

