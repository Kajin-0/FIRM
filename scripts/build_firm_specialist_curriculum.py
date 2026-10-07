#!/usr/bin/env python3
"""Build original domain-grounded conceptual and scope SFT data for compact FIRM.

Curated engineering facts are authored here, not pasted from papers. Every record
is marked PROGRAMMATIC/EDITORIAL DRAFT until independently expert-reviewed.
This is an early supplement to numerical SFT, not a standalone research corpus.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

DOMAIN_CARDS=[
("mct_alloy","How does adjusting cadmium composition change the infrared response of HgCdTe?",
"Hg1−xCdxTe is a II–VI alloy semiconductor. Adjusting x changes its band structure and bandgap and thereby the ideal absorption edge. The relationship also depends on temperature and on the empirical gap model; a measured detector cutoff is not always identical to a sharp bulk absorption edge."),
("czt_compatibility","What makes CdZnTe useful as a substrate for epitaxial HgCdTe?",
"Cd1−yZnyTe can be chosen with a lattice parameter close to HgCdTe, reducing misfit strain compared with many alternatives. It is an established HgCdTe growth substrate. Substrate orientation, crystal quality, compositional matching, defects, availability and cost still matter."),
("insb_identity","What compound does InSb denote and why does its narrow gap matter?",
"InSb is indium antimonide, a binary III–V compound semiconductor with a direct narrow bandgap, approximately 0.17 eV near 300 K. Its small gap permits mid-infrared photodetection, but thermal carriers often motivate cryogenic operation."),
("mct_mw_lw","What makes a narrow-gap photodiode suitable for MWIR versus LWIR?",
"A sufficiently small energy gap allows absorption of lower-energy photons. Choosing a longer cutoff generally increases thermal generation and dark-current challenges at fixed operating temperature. Actual performance also depends on carrier transport, device structure, passivation and spectral optical coupling."),
("photoconductor_vs_diode","Contrast infrared photoconductors and photodiodes.",
"A photoconductor detects illumination-induced conductivity changes under bias and can exhibit photoconductive gain, bias-dependent noise and contact effects. A photodiode uses its junction electric field and produces current or voltage under photovoltaic or reverse-bias conditions. They differ in impedance, noise mechanisms and bandwidth; neither is universally superior."),
("johnson_assumptions","Under what assumptions does Johnson–Nyquist noise follow 4kTR?",
"For a resistor in thermal equilibrium in the classical low-frequency limit, the one-sided open-circuit voltage noise PSD is 4 kB T R in V²/Hz and ASD is sqrt(4 kB T R) in V/sqrt(Hz). Voltage RMS noise additionally depends on effective noise bandwidth."),
("asd_psd_distinction","Explain the difference between voltage PSD, ASD and band-integrated RMS noise.",
"Voltage PSD S_v(f) has units V²/Hz; ASD sqrt(S_v) has units V/sqrt(Hz). The integrated RMS voltage is sqrt(integral S_v(f)|H(f)|² df), which depends on the transfer function and effective noise bandwidth. Mixing these definitions leads to incorrect detector noise comparisons."),
("shot_vs_johnson","How can shot noise differ from thermal resistance noise?",
"Ideal Poissonian current shot-noise ASD is sqrt(2q|I|) A/sqrt(Hz); equilibrium Johnson current-noise PSD is 4kBT/R A²/Hz for a resistor. Their assumptions differ, and measured excess, flicker, generation–recombination or readout noise can dominate."),
("noise_flicker","Why does 1/f noise matter in biased infrared photoconductors?",
"Flicker noise generally increases at lower frequencies and may depend strongly on bias and materials, contacts or interfaces. A frequency-resolved PSD and bias/temperature sweeps help separate candidate mechanisms. A single measurement point cannot establish a trap origin."),
("noise_gr","How can generation–recombination noise appear spectrally?",
"A simple single-time-constant fluctuator can yield a Lorentzian PSD with a frequency-dependent corner. Real detectors may have several lifetimes, traps and broad distributions, so a single fitted corner is not definitive proof of one microscopic recombination mechanism."),
("enbw","Why must effective noise bandwidth be used in lock-in measurements?",
"A lock-in low-pass filter weights a noise spectrum; its equivalent noise bandwidth depends on filter order and time constant. White ASD times sqrt(ENBW) estimates output RMS noise when the ASD is flat. The lock-in's display bandwidth is not automatically its noise ENBW."),
("responsivity","What does current responsivity in A/W represent?",
"Current responsivity is photocurrent divided by a consistently defined optical power. For ideal unity-gain quantum efficiency eta, R_i=eta q lambda/(hc). It depends on wavelength, collection efficiency, optical coupling, gain and operating bias."),
("nep_definition","Distinguish spectral noise-equivalent power from integrated RMS NEP.",
"Spectral NEP is noise ASD divided by matching responsivity, with units W/sqrt(Hz). An integrated RMS noise-equivalent power has units W and depends on the measurement bandwidth and transfer function. D* formulas must use the correct NEP convention."),
("dstar_units","Explain why specific detectivity requires careful area units.",
"For spectral NEP in W/sqrt(Hz), D*=sqrt(active area in cm²)/NEP_density, reported in Jones (cm sqrt(Hz)/W). The area must refer to the appropriate active/sensitive region, not automatically the package size. Converting m² or mm² incorrectly can introduce orders-of-magnitude errors."),
("dstar_limits","Can detectivity numbers from different infrared papers be compared directly?",
"Only with matching definitions and conditions. Compare wavelength, temperature, background, active area, electrical bandwidth, bias, responsivity convention and whether noise is measured or inferred. A single quoted peak D* may not describe usable imaging performance."),
("bandgap_cutoff","How does a measured infrared cutoff relate to bandgap?",
"A first approximation equates bandgap energy to photon energy Eg≈hc/lambda_c. The measured cutoff also depends on criterion (e.g. 50% spectral response), temperature, absorption tails, composition gradients and device optics. State the cutoff definition before inferring alloy composition."),
("czt_orientations","Why is crystallographic orientation important when growing HgCdTe on CdZnTe?",
"Surface orientation affects reconstruction, growth kinetics, defect incorporation and subsequent processing. Several orientations are used, including (211)B for some MBE processes. No one orientation can be declared optimal without specifying growth method and device design."),
("mbe_vs_lpe","Compare liquid-phase and molecular-beam epitaxy for HgCdTe.",
"LPE grows material from a liquid solution near equilibrium and has established bulk/epilayer uses. MBE supplies controlled atomic/molecular beams in vacuum and supports precise multilayer doping and composition designs. Tradeoffs include throughput, interfaces, defects and process controls."),
("anneal_hg","Why can mercury partial pressure matter during HgCdTe annealing?",
"Mercury chemical potential influences vacancy populations and defect equilibria; uncontrolled Hg loss can change material composition or electrical properties. Time, temperature, encapsulation, reservoir and prior history matter. Anneal recipes cannot be transferred blindly between structures."),
("mct_passivation","Why are HgCdTe device surfaces passivated?",
"Surface and interface states can contribute to leakage, carrier recombination, instability and low-frequency noise. Passivation aims to stabilize electrically and chemically sensitive surfaces. Success requires electrical, spectral and noise verification, not simply appearance or film thickness."),
("czt_lattice_mismatch","What are the consequences of lattice mismatch between HgCdTe and a foreign substrate?",
"Mismatched lattice parameters may generate misfit and threading dislocations, strain and recombination centers. Thermal-expansion mismatch adds cooldown stress. Buffer growth, substrate treatment and defect management can mitigate but do not eliminate these mechanisms."),
("foreign_substrates","What should be evaluated for HgCdTe growth on silicon or GaAs?",
"Silicon and GaAs can be alternative substrates with suitable buffer layers, but lattice/thermal mismatch, interface chemistry and defect propagation differ from CZT. Compare morphology, dislocation density, electrical performance, yield and integration goals before choosing."),
("qwp","How does a quantum-well infrared photodetector differ from a bulk HgCdTe device?",
"QWIPs use engineered intersubband absorption in semiconductor quantum wells and have optical polarization/selection-rule constraints. Bulk HgCdTe photon detectors mainly use interband absorption near the composition-dependent gap. Operational wavelength, coupling and cooling requirements differ."),
("t2sl","Why are type-II superlattices investigated for infrared detectors?",
"Type-II heterostructures spatially separate electron and hole states and can engineer effective gaps and transport by layer thickness/composition. They may offer manufacturing or dark-current advantages in some regimes, but require controlled interfaces and careful comparison against mature HgCdTe."),
("bolometer","How is a thermal bolometer different from a photon detector?",
"A bolometer responds to absorbed power via a temperature-dependent property, often resistance. A photon detector uses direct photoexcitation of electronic states. Bolometer bandwidth, thermal isolation, absorber coupling and operating temperature drive system performance."),
("darkcurrent","Why does reducing detector temperature often improve narrow-gap infrared diode performance?",
"Lower temperature can suppress thermally activated carrier generation and some leakage processes, improving signal-to-noise. It may also change freeze-out, lifetime, contact properties or optical backgrounds. Dark current is mechanism-dependent, not always a single Arrhenius law."),
("roic","What does a readout integrated circuit contribute to an infrared focal-plane array?",
"The ROIC biases, integrates, multiplexes and processes signals from detector pixels. Its capacitance, gain, read noise, well capacity, nonlinearity and frame timing can limit system performance even when detector material is excellent."),
("lockin","How should a chopped blackbody and lock-in amplifier be used to estimate detector responsivity?",
"Relate the defined optical modulation amplitude at the active detector to the demodulated electrical signal using consistent RMS/peak and harmonic conventions. Calibrate spectral transmission, source geometry, detector bias, lock-in reference and time constant."),
("blackbody_planck","Why is blackbody radiation central to infrared detector calibration?",
"Planck's law predicts temperature-dependent spectral radiance from an ideal thermal emitter. Real sources require emissivity, aperture, view factor, transmission and spectral-response corrections. Brightness temperature is not interchangeable with physical temperature without assumptions."),
("ftir","What does an FTIR interferogram represent?",
"An interferogram records optical interference versus path difference; Fourier transformation yields a spectrum after phase, sampling and apodization treatment. Resolution depends on path difference and analysis convention; source/reference correction is often required."),
("hall","What limitations affect single-carrier Hall measurements?",
"In a uniform single-carrier conductor R_H≈±1/(nq) and mu_H≈sigma|R_H|. Multicarrier transport, Hall factor, contacts, geometry, fields and inhomogeneity can invalidate the simple extraction. Report temperature and field conditions."),
("contacts","How can metal contact effects distort HgCdTe electrical characterization?",
"Nonohmic contacts, contact resistance, barriers and self-heating can alter I–V curves and inferred resistance or noise. Use contact geometry and methods such as four-terminal measurement where appropriate; check bias and temperature dependence."),
("tau_response","When does f3dB equal 1/(2πtau) for an infrared detector?",
"Only for a single-pole response with the time constant tau. Electronics poles, diffusion, trapping and thermal dynamics can add structure. Measure response versus modulation frequency and independently characterize readout bandwidth."),
("fourier","Why does Fourier analysis matter for infrared detector noise measurements?",
"Fourier-domain PSD resolves frequency-dependent noise components and spectral lines. Windowing, leakage, averaging, sample rate, anti-aliasing and normalization influence the estimator. Always report one- versus two-sided conventions and units."),
("optical_cavity","How can photonic resonances enhance thin infrared absorbers?",
"Resonant structures can increase the optical field or dwell time in a thin absorber, recovering absorption with less electrically active volume. Optical bandwidth, angle, fabrication tolerance, recombination and parasitic absorption must be co-optimized."),
("anti_reflection","Why use anti-reflection coatings on infrared detectors?",
"Index mismatch causes Fresnel reflection and reduces optical coupling. Thin-film coatings tailor interference to suppress reflection over selected bands/angles. Substrate index, material loss, thermal stability and bandwidth determine actual performance."),
("cryogenic_system","What system-level changes accompany a cryogenically cooled infrared focal plane?",
"Cooling can lower dark current but requires a cryostat or cooler, thermal isolation, cold shield, optical windows, vacuum and mechanical/electrical integration. Cooler vibration, parasitic radiation and packaging may govern achievable sensitivity."),
("netd","How should NETD be interpreted for an infrared imaging system?",
"Noise-equivalent temperature difference is the target-temperature difference corresponding to unit signal-to-noise under specified optics, scene spectrum, integration, calibration and temporal/spatial processing. It is a system metric, not a substitute for standalone detector D*."),
("arrhenius","How can temperature-dependent dark current be used to investigate activation mechanisms?",
"An Arrhenius-style plot can estimate an effective activation energy over an appropriate range, but multiple transport paths, tunneling and background contributions can confound interpretation. Test bias, temperature span and fit residuals before naming a process."),
("spectral_noise","Why is quoting one noise-density frequency sometimes misleading?",
"Detector voltage or current ASD may include 1/f, generation–recombination, white thermal and readout components. A single frequency cannot establish wideband noise or integrated RMS. Acquire a spectrum across the operating band and state bias/temperature."),
("modulation","What should be considered when interpreting chopped infrared illumination?",
"Chopper waveform duty cycle, harmonic amplitude, beam alignment, source spectrum, and lock-in RMS/peak convention determine measured signal. A chopped blackbody does not automatically supply a monochromatic power or a DC-equivalent RMS amplitude."),
("resistance_geometry","How does detector geometry affect resistance?",
"For a uniform ohmic bar R=rho L/A, where L is electrode separation and A the conduction cross-section. Contacts, current spreading, nonuniform films, semiconductor interfaces and photoconductive carrier profiles can make a naive slab estimate inaccurate."),
("photogain","What physical limitations affect photoconductive gain estimation?",
"The drift approximation G≈tau/tr=(mu V tau)/L² assumes uniform field, stable lifetime, ohmic contacts and no saturation. Recombination, trapping, field dependence and device geometry can invalidate it."),
("quantum_efficiency","Why is high optical absorption not identical to high external quantum efficiency?",
"Absorption measures photons removed from the incident beam; external quantum efficiency depends on carriers collected per incident photon. Reflection, parasitic absorption, recombination, diffusion and collection barriers reduce external QE even when total absorption is high."),
("saturation","What causes saturation and nonlinearity in infrared detectors?",
"Large optical signals may fill integration wells, cause carrier recombination/space-charge changes, self-heating, gain compression or amplifier rail limits. Test multiple irradiances and bias/readout settings rather than assuming saturation originates in the detector absorber."),
("traps","How would you distinguish interface traps from bulk recombination effects?",
"Neither mechanism can be uniquely assigned by a single transient or PSD spectrum. Compare processing variants, temperature, bias, geometry, passivation, frequency response and independent interface-sensitive measurements; report competing hypotheses."),
("spectroscopy_units","Explain why wavelength and wavenumber spectra require a Jacobian when comparing density values.",
"Spectral density per wavelength and per wavenumber refer to different bin widths. When changing variables, conserve integrated radiometric power using the absolute derivative between variables. The peak location and numerical density cannot be relabeled without transformation."),
("narrowgap_fermi","Why can nondegenerate semiconductor approximations fail in narrow-gap materials?",
"High doping or carrier densities and small bandgaps can place quasi-Fermi levels near or inside bands. Maxwell–Boltzmann statistics may then be inaccurate; Fermi–Dirac statistics, nonparabolicity and band-structure details may matter."),
("optic_etendue","What is étendue and why does it matter for infrared instruments?",
"Étendue measures area–solid-angle throughput (with refractive-index conventions) and constrains optical coupling and radiance conservation. Detector FOV, aperture, cold stop, throughput and background power depend on the actual optical geometry."),
("uncertainty","How should numerical results in infrared detector reports handle uncertainty?",
"Propagate calibration, geometric, noise-estimator, responsivity and repeatability uncertainties; separate random from systematic effects and account for correlations. Report assumptions and appropriate significant figures, not excessive decimal precision."),
("measurement_false_causes","Why should a scientist avoid inferring the exact noise mechanism from one spectrum alone?",
"Different physical mechanisms can produce similar fitted spectra over limited bandwidth. Distinguishing contacts, traps, generation–recombination and electronics requires independent controls such as bias, temperature, geometry and readout baselines."),
]

OUTSIDE_TOPICS=[
("celebrity biographies","Summarize the life of a famous television celebrity."),
("recipe preparation","Develop a weeknight pasta dish with ingredients."),
("romance","Write a dating profile introduction."),
("sport tactics","Explain when to use a high defensive line in soccer."),
("fashion","Recommend an outfit for a wedding guest."),
("marketing","Produce a slogan for a soft-drink launch."),
("vehicle repairs","Diagnose a clicking sound in my sedan dashboard."),
("poetry","Compose a sonnet about a beach sunset."),
("history","Give an overview of medieval European monarchs."),
("music","Teach me basic jazz chords on the piano."),
("investing","Choose three mutual funds for retirement."),
("social media","Write a funny caption for a pet photo."),
("fitness","Create a bodyweight exercise plan."),
("real estate","Suggest neighborhoods to buy an apartment."),
("travel","Design a weekend sightseeing itinerary."),
("politics","Argue for one contemporary political party."),
("gaming","Help beat the final level of a role-playing game."),
("gardening","Tell me when to prune hydrangeas."),
("cooking","Provide instructions to cook an omelet."),
("book reviews","Review a crime mystery book."),
("interior design","Redesign a living room with coastal furniture."),
("language learning","Give me exercises for French conversation."),
("pet care","Suggest toys for a house cat."),
("entertainment","Rank popular reality television programs."),
("restaurant advice","Recommend dishes at a sushi restaurant."),
("insurance","Compare home insurance plans."),
("tax planning","Calculate my income tax return."),
("law","Write a lease agreement for my tenant."),
("medical diagnosis","Diagnose a persistent headache."),
("nutrition","Propose a general weight-loss diet."),
("general software","Write a restaurant table-booking database schema."),
("video editing","Plan a short promotional movie for a bakery."),
("consumer hardware","Which brand of kitchen blender should I buy?"),
("retail discounts","Find the cheapest gaming console price."),
("relationship advice","Tell me how to apologize after an argument."),
("education unrelated","Explain the causes of ancient empire collapse."),
("economics","How should a coffee shop price its products?"),
("arts","Discuss Renaissance portrait painting techniques."),
("animal facts","Tell me about the mating habits of dolphins."),
("geography","List mountain ranges in Central Asia."),
("celebrity news","What's happening with a movie star's family?"),
("architecture","Design a traditional two-story house plan."),
("productivity","Give a morning productivity routine."),
("childcare","Choose bedtime stories for a toddler."),
("home appliance","How do I maintain a washing machine?"),
("cosmetics","Describe a skincare regimen."),
("general programming","Implement a chat application for restaurant patrons."),
("general space history","Give a biography of a famous astronaut."),
("fiction","Write the opening page of a detective novel."),
("social etiquette","How should I behave at a formal dinner?")
]

def build(out:Path,train_per_card:int=7,valid_per_card:int=1,train_per_out:int=6,valid_per_out:int=1,seed:int=20261006):
    if out.exists() and (out/"manifest.json").exists():
        raise FileExistsError("Dataset already materialized: use a new output directory for changes")
    out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed)
    # Independent outer prompt wrappers, with no exact user-stress evaluation text.
    wrappers=[
        "For an infrared device-engineering discussion, {q}",
        "Give a technically careful answer: {q}",
        "A researcher asks: {q}",
        "In semiconductor device physics, {q}",
        "Explain concisely with assumptions: {q}",
        "Please answer this scientifically: {q}",
        "Address this detector-research question: {q}",
        "What is the correct technical interpretation? {q}",
    ]
    out_wrappers=[
        "Could you help with this? {q}",
        "I have an unrelated request. {q}",
        "Please answer the following: {q}",
        "I am asking you directly: {q}",
        "As a chatbot, do this for me: {q}",
        "Provide a helpful response to: {q}",
        "Take your time: {q}",
        "Instead of physics, please do this: {q}",
    ]
    records={"train":[],"valid":[]}
    for i,(topic,q,a) in enumerate(DOMAIN_CARDS):
        for split,n in (("train",train_per_card),("valid",valid_per_card)):
            for j in range(n):
                offset=(j+i*3)%len(wrappers)
                prompt=(wrappers[offset] if split=="train" else "Provide a rigorous scientific explanation for this topic: {q}").format(q=q)
                # Explicit source status and expert review debt; not an asserted primary citation.
                row={"id":f"concept_{split}_{i:03d}_{j:02d}","family":topic,
                     "category":"domain_conceptual","messages":[
                    {"role":"user","content":prompt},
                    {"role":"assistant","content":a}],
                    "provenance":{"kind":"original_editorial","review_status":"awaiting_independent_expert_audit",
                                  "training_rights":"original","source_key":"curated_physics_card_v1"},
                    "training_eligible":False}
                records[split].append(row)
    refusal=("FIRM specializes in infrared photonics and related semiconductor "
              "science. That request is outside my scope.")
    for i,(topic,q) in enumerate(OUTSIDE_TOPICS):
        for split,n in (("train",train_per_out),("valid",valid_per_out)):
            for j in range(n):
                offset=(j+i*5)%len(out_wrappers)
                prompt=(out_wrappers[offset] if split=="train" else "For this task outside infrared science: {q}").format(q=q)
                row={"id":f"refuse_{split}_{i:03d}_{j:02d}","family":topic,
                     "category":"out_of_scope","messages":[
                    {"role":"user","content":prompt},
                    {"role":"assistant","content":refusal}],
                    "provenance":{"kind":"original_editorial","review_status":"scope_label_needs_review",
                                  "training_rights":"original","source_key":"scope_card_v1"},
                    "training_eligible":False}
                records[split].append(row)
    digests={}
    all_questions=set()
    for split,rows in records.items():
        rng.shuffle(rows)
        for row in rows:
            p=row["messages"][0]["content"]
            if p in all_questions:raise ValueError("Duplicate prompt across splits")
            all_questions.add(p)
        file=out/(split+".jsonl")
        file.write_text("".join(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n" for row in rows))
        digests[split]={"rows":len(rows),"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
    manifest={"schema_version":"1.0","generator":"build_firm_specialist_curriculum.py",
              "domain_cards":len(DOMAIN_CARDS),"outside_scope_cards":len(OUTSIDE_TOPICS),
              "outputs":digests,
              "training_status":"BLOCKED until independent domain fact review + scope audit; generated rows training_eligible=false",
              "content_origin":"original editorial physics principles and templated refusals, not scraped full texts",
              "limitations":["Prompt wrappers are templated; examples strongly correlated within cards",
                             "Does not by itself train a research-grade model",
                             "Adversarial scope release benchmark must remain independent",
                             "No emergency cases: medical emergencies require a separate safe exception policy"]}
    (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("data/processed/firm3_specialist_concepts_v1"))
    a=p.parse_args()
    build(a.out)
