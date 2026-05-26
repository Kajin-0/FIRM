# FIRM Dataset Rewrite Rules

FIRM should not be trained as a generic Q&A bot. It should be trained as a compact expert system for infrared photodetector physics, measurement, fabrication, and empirical troubleshooting.

## Core answer standard

A high-quality FIRM answer should usually include:

1. **Assumptions**: temperature, bias mode, detector type, geometry, bandwidth, optical loading, or material system when relevant.
2. **Governing equation**: the minimal equation needed to anchor the explanation.
3. **Variable definitions and units**: avoid unitless handwaving in detector calculations.
4. **Physical mechanism**: connect the equation to carrier generation, transport, recombination, optical absorption, or measurement electronics.
5. **Measurement implication**: explain how the effect appears in IV, responsivity, noise PSD, frequency response, transmission, Hall data, or blackbody testing.
6. **Caveat**: state when a conclusion depends on composition, temperature, geometry, passivation, readout, or calibration.

## Redundancy policy

The original dataset contains many repeated numeric template families. These are useful for arithmetic grounding but harmful if overrepresented.

### Keep

- 3-5 examples of generic plug-and-chug formula families.
- 10-20 examples of high-value detector formula families when they vary in interpretation.
- Rows that teach units, assumptions, or measurement context.
- Rows that connect a calculation to detector performance.

### Compress or remove

- Hundreds of nearly identical dopant-ionization rows.
- Many repeated NEP/D* examples that differ only numerically.
- Repeated shot-noise examples without interpretation.
- Short beginner rows that teach generic semiconductor facts without infrared relevance.

## Rewrite policy

### Weak row

```text
What is dark current?
Dark current is current without light. It adds noise.
```

### Strong row

```text
Dark current is the detector current that flows with no target optical signal. In HgCdTe it is strongly tied to bandgap, temperature, carrier concentration, junction or surface leakage, and bias. It contributes shot noise, G-R noise, reduced dynamic resistance, and lower D*. It should be reported with temperature, bias, geometry, and bandwidth context.
```

## Required distinction rules

FIRM must distinguish:

| Pair | Required distinction |
|---|---|
| Responsivity vs D* | Signal conversion vs noise-normalized sensitivity |
| Photocurrent vs dark current | Modulated useful signal vs non-signal current/noise source |
| Johnson noise vs shot noise | Thermal resistance noise vs carrier-flow discreteness |
| 1/f noise vs G-R noise | Power-law low-frequency noise vs Lorentzian lifetime/trap process |
| Cutoff wavelength vs composition | Optical transition estimate vs material-state inference requiring temperature/model |
| Detector bandwidth vs lock-in filter bandwidth | Device physics vs measurement-chain filtering |
| IV quality vs optical response | Electrical conduction test vs complete photoresponse behavior |
| Material quality vs measurement artifact | Physical inference must survive controls and geometry scaling |

## Preferred mathematical style

Use compact equations when they clarify:

```text
E_g[eV] ≈ 1.2398 / λ_c[µm]
n_i ∝ exp(-E_g / 2k_B T)
v_J = sqrt(4 k_B T R Δf)
e_J = sqrt(4 k_B T R)
NEP = v_n / R_V
D* = sqrt(A Δf) / NEP
G ≈ τ / τ_t
τ_t = L^2 / (µV_b)
f_3dB = 1 / (2πτ)
S_GR(f) = S_0 / [1 + (2πfτ)^2]
```

Always define whether a noise value is RMS over bandwidth or spectral density.

## Empirical inference policy

FIRM should avoid overclaiming from one measurement.

Use:

- **consistent with**
- **suggests**
- **rules against**
- **requires additional measurement**

Avoid:

- definitive defect attribution from one IV curve;
- claiming material quality from responsivity alone;
- extracting microscopic lifetime from one lock-in roll-off without readout/source de-embedding;
- giving HgCdTe composition without temperature, cutoff criterion, and bandgap model.

## Dataset construction policy

The strongest FIRM dataset should be a mixture:

| Type | Target share |
|---|---:|
| Expert derivations | 25-35% |
| Empirical troubleshooting | 20-30% |
| Measurement workflows | 15-20% |
| Compact formulas/calculations | 10-20% |
| Definitions / grounding | 5-10% |
| Guardrails / uncertainty discipline | 5-10% |

The original dataset was strong for grounding but too short-answer-heavy. FIRM v2 should bias toward deep, analytical, empirical, and mathematical reasoning.
