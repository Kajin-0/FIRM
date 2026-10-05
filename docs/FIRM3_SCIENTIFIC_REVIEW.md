# FIRM 3 scientific review — 2026-10-05

Review authority: Codex independently derived calculations and individually inspected the selected answers. This is recorded AI analytic review, not human expert signoff. Original sources, historical candidate v2 and all published eval releases are unchanged.

## Eight-case pilot

All eight v1 cases were independently re-derived without calling the original generator. Newton root finding, Decimal Planck calculation, alternate transport substitutions, log-frequency quadrature and Gamma-function ENBW agree with all 22 stored quantities to better than 1e-10 relative difference. V1 remains unchanged; no v2 is required.

| Case | Independent result(s) | Scientific checks |
|---|---|---|
| numeric_pilot_001 | gap_85K=0.13189808 eV; composition_x=0.22885389 1; cutoff_150K=8.2241473 um | Supplied empirical fit only; temperature fixed for root; cutoff criterion is not measured responsivity cutoff. |
| numeric_pilot_002 | nep_before=2.5e-12 W/sqrt(Hz); nep_after=8.5714286e-13 W/sqrt(Hz); dstar_before=2.5298221e+10 Jones; dstar_after=7.3786479e+10 Jones; dstar_ratio=2.9166667 1 | 0.40 mm² = 0.004 cm²; density NEP needs no bandwidth factor; one frequency does not establish a trap mechanism. |
| numeric_pilot_003 | carrier_tau=0.00015 s; detector_f3db=1061.033 Hz | Cascade amplitudes multiply; Hz needs 2π; phase/source/readout controls required. |
| numeric_pilot_004 | dark_resistance=1444.7938 Ohm; transit_time=8.8888889e-07 s; gain=3.375 1 | cm^-3 → m^-3 ×1e6; cm²/Vs → m²/Vs ×1e-4; supplied independent lifetime; Hall factor unity. |
| numeric_pilot_005 | corner_frequency=44.444444 Hz; rms_noise=8.1795992e-08 V | Integrate one-sided PSD, not ASD; finite 1/f lower limit; real filters need transfer weighting. |
| numeric_pilot_006 | radiance_per_um=50.914036 W/(m^2*sr*um); modulated_power_difference=6.7206528e-10 W; voltage_difference=5.3765222e-06 V | Per-m → per-um ×1e-6; projected solid angle already includes cosine; state difference is not chop fundamental/RMS. |
| numeric_pilot_007 | thickness_95pct=29.957323 um; absorption_20um=0.86466472 1 | alpha in cm^-1 needs cm thickness; absorption does not equal external QE. |
| numeric_pilot_008 | enbw=12.5 Hz; rms_noise=4.2426407e-08 V | One-sided unity-gain pole integral 1/(4tau); post-demodulator normalization supplied. |

Numerical tolerances (rtol=0.005, atol=0) allow arithmetic rounding. They are not physical uncertainty limits for empirical material models, optics or calibration. All manual rubrics appropriately request assumptions, competing explanations and controls; subjective grading still requires reviewer calibration. Pilot 007 remains an explicit legacy regression, not unseen evidence.

## Original E1 selection and review

The old trainer takes `rows[:32]` after the candidate file was sorted by stable ID. Seed 42 controls training initialization/order, not subset selection. Source file hash and all 32 exact prompt/answer hashes are pinned in `data/reviews/firm3_manual_decisions_v1.json`.

**25 accepted, 1 corrected, 6 excluded, 0 unresolved among these 32.** “Accepted” means correct within the recorded scope; numerical shot-noise examples assume independent Poisson currents, a matched bandwidth and no omitted dominant source. Source licensing remains unknown for every selected row.

Source abbreviations: R = rewritten SFT; E1/E2 = expert HgCdTe batch 01/02. Full IDs, source hashes, family identity, scientific/numerical/unit correctness, answer quality, provenance and reasons are machine-readable in the decisions file. Table IDs/families are prefixes only.

| Row / ID | Source:line | Family | Topic | Decision / science; number; units | Quality / trainable | Reason |
|---|---|---|---|---|---|---|
| 1 / 0027c52061dd | E2:2 | 77f8e9c0 | Lock-in bandwidth and D* | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Correct density-versus-RMS/Jones conventions. Density inference from ENBW requires white or explicitly integrated PSD. |
| 2 / 003f7c07a09d | R:2008 | 76174894 | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | Photocurrent and Poisson shot RMS independently recalculated; neglect dark/background/excess noise. |
| 3 / 00802401e3d9 | R:1997 | 76174894 | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | Photocurrent and Poisson shot RMS independently recalculated; neglect dark/background/excess noise. |
| 4 / 0081ae6c1815 | R:520 | b73cb142 | Radiometry | correct; corrected; not_applicable; not_applicable | adequate anchor / yes | Corrected universal Poisson photon-statistics claim: thermal bunching and excess statistics can matter. |
| 5 / 008a7c330c51 | R:325 | f82f91e9 | Recipe Boundary | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Narrow domain redirection; no numerical or physical assertion. |
| 6 / 009174173d7f | R:702 | c4fb0d47 | Dopant Ionization | exclude; fail; not_applicable; not_applicable | adequate anchor / no | Activation energy and temperature alone do not determine dopant occupation; Fermi/neutrality/degeneracy missing. |
| 7 / 00e910d56cef | R:82 | 69631ed0 | Cost Drivers | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Bounded qualitative system-cost drivers; no invented price or universal ranking. |
| 8 / 012bfe25a6e7 | R:2123 | 76174894 | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | Photocurrent and Poisson shot RMS independently recalculated; neglect dark/background/excess noise. |
| 9 / 013f28c407a4 | R:2183 | 81992b0d | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | One-sided Poisson current variance integrated over stated bandwidth; independently recalculated. |
| 10 / 015ff2dcdf83 | R:436 | c292e788 | Humanities Boundary | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Domain boundary permits detector-related history/literature; no physics defect. |
| 11 / 016a39a96659 | R:644 | c4fb0d47 | Dopant Ionization | exclude; fail; not_applicable; not_applicable | adequate anchor / no | Under-specified ionization and unsupported freeze-out fraction. |
| 12 / 01f9459f66c9 | R:2428 | defcce71 | Optics | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Correct power-transmittance definition; specify spectral/geometry conditions in measurements. |
| 13 / 0255fc6f7dd5 | R:354 | 490b3891 | Preference Boundary | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Domain boundary statement; no physics defect. |
| 14 / 02c3a4b5355d | R:2148 | 81992b0d | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | Poisson shot RMS recalculated from displayed current and bandwidth. |
| 15 / 02cb4869e0a1 | R:475 | 4ef80f5e | NEP and D* Relationship | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Correct Jones area and density/RMS distinction; bandwidth-scaled comparison assumes matched spectral response. |
| 16 / 02f8c25be4ca | R:598 | 572f69ba | Simulation Tools | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Kinetic carrier-scattering description valid; drift-diffusion has limited high-field/nonequilibrium reach. |
| 17 / 032060723feb | R:1387 | 2c90c9cb | Semiconductors | exclude; fail; not_applicable; not_applicable | adequate anchor / no | Intrinsic ni and degeneracy model absent; 0.81 V cannot be uniquely verified from prompt. |
| 18 / 03c0622ce6aa | R:2222 | 81992b0d | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | Poisson shot RMS recalculated from displayed current and bandwidth. |
| 19 / 03d29e1bacac | R:1459 | 83aefcb8 | Quantum Mechanics | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Correct statistical mixed-state definition; basic quantum-physics anchor. |
| 20 / 03f529d69aca | R:547 | f5c6dbcb | Poisson Equation | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Poisson signs and constant-permittivity reduction correct; necessary boundary data explicit. |
| 21 / 041c06e0707b | R:434 | ed8ea788 | General Assistant Contrast | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Project identity/desired behavior; this does not establish superiority over a generic model. |
| 22 / 04546e30f7af | R:1482 | 54eb69a4 | Semiconductors | exclude; fail; not_applicable; not_applicable | adequate anchor / no | Under-specified dopant statistics; do not relabel algebra as physical ionization. |
| 23 / 0489384b1679 | R:2455 | 4bf4590c | Photodetectors | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Amplitude SNR convention accepted: signal magnitude over RMS noise in a stated band; power SNR is its square. |
| 24 / 04bcaf1f3a01 | R:1704 | 16e7d658 | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | RMS-current/response NEP recalculated; given A implies W, not W/sqrtHz. |
| 25 / 05070781ecad | E1:4 | 274b4cd3 | Generation-recombination noise | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Phenomenological Lorentzian form correct; fitted trap correlation time is not necessarily optical minority-carrier lifetime. |
| 26 / 05168c8828a2 | R:638 | c4fb0d47 | Dopant Ionization | exclude; fail; not_applicable; not_applicable | adequate anchor / no | Under-specified dopant statistics and microscopic freeze-out claim. |
| 27 / 057e4027d6dd | R:1765 | 16e7d658 | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | NEP recalculated; reported rounding differs 0.14%, acceptable for legacy displayed parameters. |
| 28 / 0592469d67f0 | R:1663 | 16e7d658 | Photodetectors | accept; pass_with_stated_scope; pass_with_0.5pct_printed_rounding; pass | adequate anchor / yes | NEP recalculated; explicit integrated units consistent. |
| 29 / 059c69ebfba4 | R:1409 | 603f5e2f | Optoelectronics | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Longitudinal-mode switching description correct; no unsupported numerical claim. |
| 30 / 05df2615010a | R:346 | cba0d6d9 | Lifestyle Boundary | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Domain boundary and useful materials context; no physics defect. |
| 31 / 06459ffc298d | R:172 | a7aa5209 | Phase-Sensitive Detection | accept; pass_with_stated_scope; not_applicable; not_applicable | adequate anchor / yes | Synchronous mixing/low-pass and calibration cautions correct; actual instrument normalization must be specified. |
| 32 / 06a8fac629ee | R:1479 | 54eb69a4 | Semiconductors | exclude; fail; not_applicable; not_applicable | adequate anchor / no | Under-specified dopant statistics; no unique fraction. |

The correction bounds the universal photon-Poisson claim: independent detected photoelectrons give one-sided SI=2qI under a Poisson model, but thermal bunching/source excess statistics can matter. Excluded rows 6, 11, 22, 26 and 32 lack Fermi level, degeneracy, carrier statistics/neutrality. Row 17 lacks an intrinsic-density/degeneracy model for its silicon junction.

Thirty additional anchors were individually inspected to replace unsafe smoke material and provide reviewed validation examples. They are separately marked in the same decisions file; no uninspected row is relabeled reviewed.

## Versioned D* correction and triage

Each of the 150 strictly recognized canonical D* prompts supplies positive A in m², bandwidth in Hz and RMS NEP in W. Every printed old result agrees within 2% with sqrt(A_m²*bandwidth)/NEP while being labeled Jones. The correction converts A to cm² *before* square root and recomputes from the printed prompt, not by blindly multiplying a rounded old answer. Old/new values, units, formula, source hash/line, old answer/hash and Jones-SI-area-v1 provenance are recorded.

`data/reviews/firm3_scientific_v1/detectivity_corrected_v1.jsonl` contains 150 new rows. No other family is transformed by this correction. White/matched-bandwidth caveats and explicit cm*sqrt(Hz)/W units are included. Originals remain byte-identical, including historical CSV representations.

The review ledger records every canonical seed row plus all flagged historical representations: 2,581 canonical rows; 2,889 ledger entries. All prior arithmetic/scientific flags and explicit measurement-model/eval-equivalence exclusions are retained. Under-specified dopant numerical families are excluded. Uninspected rows remain needs_review; the four non-D* arithmetic flags are not guessed away. Scientific acceptance, programmatic correction, unreviewed status, eval eligibility and source rights are separate gates.

Regenerate review into a new directory: `python3 scripts/review_firm_seed.py --out-dir data/reviews/new_review_version`. Source/manifests are hashed. Human review remains required before any substantive adaptation.
