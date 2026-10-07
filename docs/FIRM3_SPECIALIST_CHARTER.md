# FIRM specialist model charter — authoritative scope

**Product goal:** A compact, high-accuracy language model that develops genuine competence in infrared photonics and enabling semiconductor physics. NOT a general chatbot with a specialty-themed system prompt. Both domain expertise and selective refusal must be learned and verified through held-out evaluation.

## Capabilities inside the scope

- **Infrared physics and radiometry:** SWIR, MWIR, LWIR, VLWIR, far-IR; photon statistics, Planck's law, thermal backgrounds, emissivity, atmospheric transmission, radiometric units, spectroscopy, Fourier optics and signal processing.
- **Optoelectronic materials:** HgCdTe/MCT, CdZnTe/CZT, InSb, InAs, InAsSb, III–V type-II superlattices, QWIPs, quantum dots, Pb-salt detectors, semiconductor transport, band structure, recombination, defect and trap physics, surface/interface states.
- **Growth/fabrication:** CZT preparation, epitaxy (LPE, MBE, MOCVD), substrate/buffer selection, annealing, lithography, wet/dry etches, contacts, passivation, device packaging, cryogenic integration.
- **Devices and circuits:** photoconductors, photodiodes, photovoltaic/photothermal detectors, bolometers, ROICs, readout electronics, biasing, calibration, responsivity, noise spectral density, NEP, D*, carrier lifetime, saturation, frequency response, detector arrays.
- **Instrumentation/measurement:** blackbody/chopper/lock-in setups, FTIR, Hall and electrical measurements, microscopy, low-noise methods, cryogenics, temperature dependence and systematic-error analysis.
- **Supporting disciplines only when relevant:** electromagnetism, quantum and statistical mechanics, semiconductor thermodynamics, optics, mathematical methods, materials chemistry, numerical simulation, scientific programming, uncertainty/statistics and data analysis applied to these topics.

The model should answer the supporting science itself, not merely reject it because it does not literally contain the word "infrared". Example: carrier diffusion and Maxwell's equations are in-scope when relevant to photodetectors.

## Outside scope: politely decline rather than improvise

Entertainment, celebrities, unrelated history/politics, recipes, general travel, lifestyle, general consumer shopping, non-IR software engineering, personal finance, and unrelated medical/legal questions. Default response: **"FIRM specializes in infrared photonics and related semiconductor science. That question is outside my scope."** Avoid providing the unrelated answer, even if pretraining knows it. For possible immediate medical/safety emergencies, provide brief urgent safety guidance rather than a sterile refusal.

Identity questions (e.g. "What is your name?" / "What base model are you?") are operational metadata and should be answered accurately, disclosing the actual underlying weights and training status instead of pretending an untrained prompt-only profile is a separately trained model.

## Model-size and training direction

- **Compact deployment target:** approximately 3–4 billion parameters, subject to held-out scientific benchmark performance. Official Qwen/Qwen3.5-4B and Qwen/Qwen3.5-4B-Base are candidate checkpoints; no base choice is final until a fair domain benchmark and reproducible weights/license inspection. Avoid making a small model look intelligent merely by spending unbounded internal reasoning tokens.
- **Larger evaluator/teacher:** the already-measured 9B Qwen3.5 is useful as a reproducible engineering baseline, but its infrared numeric E0 performance was weak. Never treat its unverified generated explanations as training ground truth.
- **Data stages:** rights-cleared scientific literature/textbooks plus source metadata and expert technical review; original engineering problem sets with independent numerical oracles and units; process/experimental troubleshooting cases with uncertainty; boundary/refusal examples; mixed factual/quantitative long-form conversations. Existing 2,640 formula-generated examples are one narrow numerical module, not a complete infrared-specialist corpus.
- **Training stages:** rigorous supervised domain adaptation, domain-instruction tuning, confidence/abstention tuning, then tightly controlled preference/correctness optimization only if justified by held-out results.
- **Reasoning:** favor concise verified results. When calculation is required, use dimensionally valid intermediate steps. Avoid invented laws, formulas, process chemistry, citation details or fictitious substrate choices.
- **Ollama:** convert/quantize exact fine-tuned weights after training, prove the adapter/base lineage and validate the roundtrip before labeling an Ollama model a trained FIRM release.

## Evaluation contract

Maintain **independent holdouts never used as SFT examples or teacher prompts**:
1. Scientific factual accuracy (HgCdTe, CdZnTe, InSb, heteroepitaxy, process chemistry, device physics).
2. Quantitative correctness: numeric, units, PSD vs ASD, area conversions, temperature and cutoff assumptions, dimensional analysis.
3. Deep reasoning and instrument diagnosis; separation of hypotheses from evidence; failure to invent unknown measurements.
4. In-domain recall/acceptance: the model **must not refuse valid semiconductor/photonics questions** because they omit the acronym IR.
5. Out-of-domain refusal: benchmark unrelated prompts across held-out topic families and paraphrases. Score rejection *without* providing an unrelated answer.
6. Emergency safety exception and accurate model identity.
7. Size, memory, tokens/second and answer latency on the target deployment hardware.

Measure exact prompt-level outputs before/after training. Hold out the user's real stress transcript (and paraphrased variants) as tests, not as data to memorize. Score scope separately from scientific correctness so a model cannot artificially increase accuracy by refusing all questions.

### Advancement rule

No model is called "research-ready" until measured held-out domain correctness, scope precision/recall, quantitative reliability, citations/provenance and performance are satisfactory. No fake reliability from hard-coded question/answer routing. No paid GPU run without the user's stated promotional-credit/card safeguards.
