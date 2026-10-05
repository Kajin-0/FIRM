# FIRM 3 specification

Status: architecture and measurable interfaces established; no FIRM 3 weights trained. FIRM is intentionally specialized. Its objective is correct quantitative detector work and experimentally disciplined inference, rather than broad conversational coverage.

## Domain and target behavior

Scope includes HgCdTe/MCT, InSb, InGaAs, InAsSb, T2SL, QWIP/QCD and related SWIR/MWIR/LWIR/VLWIR materials; semiconductor statistics, transport and recombination; photoconductors, photodiodes and APDs; noise, responsivity, NEP, D*, radiometry and blackbody calculations; epitaxy, LPE/MBE/MOCVD, annealing, contacts, passivation and device processing; cryogenic, Hall, lock-in, FTIR and spectral-response measurements; ROIC/electronics fundamentals where they affect interpretation.

FIRM should solve linked problems: composition -> bandgap/cutoff -> carrier density/transport -> resistance/gain -> responsivity -> PSD/bandwidth -> NEP/D* -> experimental interpretation. It should also invert measurement evidence into competing mechanisms and discriminating experiments. A passivation-related PSD change, for example, can involve surface traps, contacts, optical transmission, bias, heating, electronics or bandwidth; one observation should not prove a microscopic mechanism.

Answers should expose relevant assumptions, equations, defined variables and units, calculation/derivation, dimensional and numerical checks, physical interpretation, measurement consequences, competing explanations, empirical limits and validation experiments. Use only the fields that help the task. Do not train a rigid ten-heading response template for every definition. Missing parameters, inconsistent data and non-identifiable mechanisms should produce a bounded answer or a specific request for the missing quantity.

Critical distinctions: one-sided versus two-sided PSD; ASD versus PSD versus integrated RMS; spectral-density NEP versus bandwidth-integrated NEP; current versus voltage responsivity; metre-based detectivity versus Jones; carrier/trap time versus detector/readout/source poles; absorption edge versus measured spectral-response cutoff; Hall mobility versus drift mobility/Hall factor; dark current versus useful modulated signal; a fitted empirical coefficient versus a material constant.

## Dataset architecture

Preserve current paths and history. Add new versioned artifacts only when a real corpus exists:

| Directory | Purpose |
|---|---|
| `raw/` | Immutable ingested files; never replace an uploaded source |
| `external/` | Rights-reviewed source documents with retrieval identities |
| `generated/` | Synthetic drafts with generator/model revision and seed |
| `processed/` | Reproducible candidate/reviewed SFT or packed DAPT outputs |
| `heldout/` | Restricted test assets; never include in wildcard training ingestion |
| `manifests/` | Schemas, file/record hashes, licensing/review state, release identities |
| `audits/` | Distributions, quality flags, contamination candidates and review decisions |

Empty directories and placeholder corpora are unnecessary. The current root CSV and processed exports stay in place. A future release should explicitly select source manifests rather than scan every file under `data/`.

Training taxonomy:

| Kind | Required capability / review emphasis |
|---|---|
| Short factual anchors | Precise, bounded definitions; avoid unsupported universal ranges |
| Conceptual explanations | Mechanisms, applicability and distinctions |
| Single-step calculations | Numeric accuracy and explicit unit conversion |
| Multi-step calculations | Linked quantities, uncertainty and end-to-end consistency |
| Full derivations | Valid assumptions, intermediate equations and limiting cases |
| Troubleshooting | Competing hypotheses, controls and falsifiable predictions |
| Measurement interpretation | Data/geometry/calibration provenance and instrument transfer |
| Detector design | Objectives, tradeoffs, constraints and discriminating validation |
| Literature-grounded examples | Exact paper/version/page, claim attribution and rights |
| Tool use | Typed call, reproducible execution, result verification and interpretation |
| Multimodal | Genuine labeled plots/images with paired underlying data where available |
| Preference/correctness | Reviewed pair, explicit defect label, no stylistic preference proxy |
| DAPT | Document/section/rights identities, extraction and split/packing lineage |

Record contract for a reviewed SFT release: stable `id`, schema version, `messages`, task kind, material/mechanism labels, difficulty, `family_id`, optional `document_id`, quantity/unit conventions, provenance and eligibility. Provenance must distinguish human-authored, synthetic/model-generated, transformed and literature-derived; include source hash/location, generator and revision where applicable, source license/permission, review status/reviewer and transformation lineage. Eligibility must explicitly distinguish train, validation, test, quarantine and exclusion. Unknown authorship or license stays `unknown`, not automatically human-authored or public domain. Current candidate metadata already records origins, hashes, groups and unreviewed status.

Target 5,000–15,000 high-quality reviewed SFT examples after a smaller pilot. Increase size only when coverage/review capacity justifies it. A provisional sampling mix is 10% short anchors, 15% conceptual, 20% linked calculations, 15% derivations, 25% troubleshooting/measurement interpretation, 10% design/literature and 5% uncertainty/domain boundaries. Tool and multimodal subsets are added separately as genuine assets become available. Correctness is a release gate, not a response-length target.

Long-form drafts may organize problem, known/required quantities, assumptions, equations/units, derivation, dimensional/numerical checks, interpretation, alternatives and experiments. Vary style and task structure. Use independently computed numerical oracles, adversarial unit perturbations, insufficient-information examples and before/after controls. Existing generic numerical families should be capped and scientifically repaired, rather than inflated with paraphrases.

## Benchmark architecture

Keep three distinct assets: frozen legacy diagnostics; a public numerical development pilot; a future frozen independent benchmark with public protocol and privately held solutions where feasible. Existing files are never repurposed into training. Public pilot answers are forbidden training inputs, even though foundational equations should be taught elsewhere. Fine-tune development uses a separate development set; repeated inspection of a fixed test consumes its evidentiary value.

Future coverage matrix:

| Area | Required tests |
|---|---|
| Material/composition | HgCdTe gap/composition/cutoff with supplied empirical model, temperature, validity and uncertainty; InSb/InGaAs/InAsSb/T2SL distinctions |
| Statistics/transport | Intrinsic and doped statistics, compensation, Hall factors, conductivity, geometry and non-identifiability |
| Device physics | Photoconductors, photodiodes, gain/transit/lifetime, dark current, collection and recombination |
| Noise/performance | Johnson, shot, GR/Lorentzian and 1/f; PSD/ASD/integration; responsivity, NEP and D* conversions |
| Dynamics/electronics | Detector, RC, source, preamp, chopping and lock-in filters; amplitude and phase de-embedding |
| Radiometry | Planck spectral conventions, blackbody integrals, throughput/etendue, emission/reflection, calibration and uncertainty |
| Characterization | FTIR/spectra, temperature/Arrhenius sweeps, IV, Hall, noise and cryogenic controls |
| Growth/process | LPE/MBE/MOCVD, annealing, contacts, passivation and geometry comparisons with bounded evidence |
| Research/design | Citation-backed literature interpretation and linked multi-step design tradeoffs |

Start with tens of expert-reviewed independent scenarios, then cover missing areas; do not set a thousands-of-questions quota. Use family/document grouping, reserved scenarios, dimensional variants, controlled distractors, inconsistent-data cases and ablations. Identical tasks with changed numbers belong to one family and one partition. Retrieve/synthesize training data without access to permanent hidden solutions.

Machine-readable schema: `evals/firm_benchmark_schema.json`. Each item includes version/ID/category/family, prompt, provenance/answer-publicness, structured quantity oracles with units/tolerances and manual rubric. Versioned manifests fix source/file/record hashes, generation date, disclosure status and leakage-check history. Any prompt, answer or tolerance change requires a new release identity. A leakage candidate is investigated and quarantined, not silently dismissed or rewritten away.

| Metric | Implemented or planned meaning |
|---|---|
| Numerical accuracy | Implemented: declared quantities within fixed absolute + relative tolerance; missing quantities fail |
| Relative numerical error | Implemented per quantity after supported unit conversion; undefined for zero gold values |
| Unit correctness | Implemented for an explicit limited conversion vocabulary; unknown units fail and can later be reviewed |
| Physical/equation correctness | Human-reviewed applicability and dimensional/physical validity; currently unscored |
| Derivation validity/completeness | Expert rubric for necessary intermediate steps, checks and assumptions |
| Diagnosis quality | Hypothesis coverage, evidence discrimination and useful validation experiments |
| Citation correctness | Claim-to-source/page entailment, paper/version identity, quotation fidelity |
| Unsupported-claim rate | Reviewer counts claims requiring evidence and unsupported assertions; no reliable automatic proxy yet |

Do not collapse these into an opaque single score. Report coverage, category distributions, missing predictions, truncation and manual-review status. Keyword coverage is retained only as a diagnostic. For release comparisons, fix benchmark, model revision, system/template, tool/retrieval policy, token budget, sampling settings and quantization. Report no-tool and tool-enabled tracks separately. Use paired results and family-level uncertainty when enough independent scenarios exist; 8 pilot cases cannot establish broad superiority.

## Scientific tools

Begin with a small provider-independent interface, not an autonomous agent framework. A tool request carries call ID, tool/version, typed operation, input values/units or dataset hash, assumptions and seed. A result carries status, outputs/units, warnings, code/environment version, input/output hashes and execution limits. The model must check units and limiting cases before interpreting the result. Runtime tools are not implemented yet.

Initial operations: calibrated Planck integration; gap-model inversion; noise conversion/integration; Hall/geometry calculations; transfer-function fitting; constrained nonlinear least squares and fit diagnostics; uncertainty propagation/Monte Carlo; reproducible plots. Use Python/NumPy/SciPy/pandas, with SymPy only for tasks needing symbolic work. Curve fitting must preserve uncertainties, weighting, parameter bounds, residuals and identifiability. SPICE and FEniCSx are later additions when a real task needs them.

Expose named operations first. If arbitrary Python is later supported, isolate it with time/memory limits, explicit mounted inputs and no network/secrets. Treat files, papers and plot annotations as data rather than executable instructions. Training examples must include correct calls, unit mistakes, rejected inputs, solver failures, insufficient data, corrected execution and honest uncertainty. Scientific tools should verify calculations, not invent empirical constants.

## Literature-grounded retrieval

Parametric learning: detector/statistical/transport physics, mechanisms, common equations, radiometry, measurement principles and reasoning methods. Retrieval: exact literature values, material-specific fit coefficients, fabrication conditions/recipes, manufacturer specs, recent findings and proprietary documents within authorized access.

Document identity includes DOI/report ID, title/authors, version/date, source URL, content hash and rights/access class. Extraction preserves page/section coordinates, tables/captions, equation IDs and quotation boundaries. Retrieval returns evidence IDs, scores, query, document/version/page and exact excerpt spans, not anonymous chunks.

Prototype with a small rights-reviewed local corpus: metadata/material/temperature filters, lexical retrieval plus an optional measured embedding improvement, equation/table-aware sections and small neighbor context. Rerank only if a benchmark shows value. Build a claim-to-evidence ledger for material-specific claims; label inference versus measured results, reconcile conflicting conditions and abstain when sources do not support the claim. Preserve numerical units, source uncertainties and cutoff/noise definitions. Evaluate retrieval recall, citation entailment, unsupported claims, conflicting-source handling and permission boundaries separately from model reasoning. Do not ingest evaluation solutions or expose private documents through citations.

## Multimodal roadmap

M0: analyze real tabular IV/PSD/spectral/temperature/Hall/frequency data through tools, independent of a vision encoder. M1: read genuine plotted curves with verified axes/units, log scales, legends, error bars and paired numeric data. M2: diagnose correlated plots and measurement setups. M3: layer diagrams/schematics, SEM and microscopy with expert labels and limits on inferring composition/defects from images alone.

For each asset record creator/license, original file hash, instrument/acquisition settings, device/sample ID, underlying data and split family. Samples, wafers, papers and repeated images stay in one partition. A native vision-language candidate enables experimentation, not a claim of trained IR visual expertise. Preserve vision encoders during initial text SFT and evaluate degradation; only fine-tune visual components when sufficient real labeled data exist. No fake SEMs, invented experimental plots or unlabeled screenshots are counted as capability data.

## Success gates

Data: no unresolved exact/template eval overlap in eligible release rows; record provenance and review coverage; document semantic-audit limitations. Physics: independently verified quantity oracles, no persistent unit/density errors, explicit non-identifiability and reliable controls. Model: paired improvement over its unmodified checkpoint on independently reviewed tasks, no material regressions in protected categories, and practical latency/memory. Deployment: adapter/base revision traceability, reproducible export, quantization regression tests and local validation. Numerical targets for the permanent benchmark are set after E0 and expert rubric calibration; no baseline score is fabricated now.
