#!/usr/bin/env python3
"""Build FIRM-4B v2 foundational specialist curriculum.

Original, deterministic training data only. No private benchmark prompts/gold are read.
Goals: canonical IR facts, domain taxonomy, typo robustness, natural quantitative
reasoning, selective refusal, and empirical uncertainty discipline.

All factual cards are authored as compact claims with evidence metadata. They remain
PROVISIONAL_RESEARCH until independent expert audit; source URLs support claims but
source text is not copied into training answers.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, re
from pathlib import Path

HC=1.2398419843320026
K=1.380649e-23
Q=1.602176634e-19

SYSTEM=("You are FIRM, the Focused Infrared Research Model: a compact specialist in "
        "infrared photonics, semiconductor detector materials/devices, fabrication, "
        "characterization, radiometry, noise and supporting physics. Answer in-scope "
        "questions directly. State assumptions and units. Do not invent material formulas, "
        "measurements or mechanisms. Decline wholly unrelated requests.")

SOURCES={
"czt_hgcdte":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12944146/",
"czt_match":"https://pmc.ncbi.nlm.nih.gov/articles/PMC9533753/",
"ingaas":"https://www.hamamatsu.com/us/en/product/optical-sensors/infrared-detector/ingaas-photodiode/G17190-005K.html",
"ir_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7282310/",
"mid_ir_platform":"https://pmc.ncbi.nlm.nih.gov/articles/PMC5090602/",
"mid_ir_sources":"https://pmc.ncbi.nlm.nih.gov/articles/PMC9420685/",
"qcl_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12552865/",
"microbolometer":"https://pmc.ncbi.nlm.nih.gov/articles/PMC12430745/",
"roic_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC10007274/",
"chalcogenide":"https://pmc.ncbi.nlm.nih.gov/articles/PMC6270612/",
"hot_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC10490682/",
}

# (id, question, answer, evidence key)
FACTS=[
("mct_identity","In infrared detector engineering, what does MCT mean?",
 "MCT means mercury cadmium telluride, Hg₁₋ₓCdₓTe (HgCdTe), a ternary II–VI semiconductor alloy. Its composition-dependent narrow bandgap is why it is widely used for infrared photon detectors. In a FIRM/IR context, interpret MCT this way unless the user clearly specifies another meaning.","ir_review"),
("czt_identity","What is CZT in HgCdTe detector work?",
 "CZT means cadmium zinc telluride, Cd₁₋ᵧZnᵧTe. It is a II–VI semiconductor commonly used as a substrate for HgCdTe because its lattice parameter can be closely matched to HgCdTe. CZT is also used separately as a room-temperature X-ray/gamma-ray detector material; those are different applications.","czt_hgcdte"),
("czt_role","Why is CdZnTe used as a substrate for HgCdTe?",
 "CdZnTe is the conventional near-lattice-matched substrate for HgCdTe epitaxy. Good lattice matching reduces misfit strain and dislocation formation compared with strongly mismatched foreign substrates. Zn content, orientation, substrate defects, cost and available wafer size still matter.","czt_match"),
("insb_identity","What is InSb?",
 "InSb is indium antimonide, a binary III–V semiconductor with a very narrow direct bandgap of about 0.17 eV near 300 K. It is an important infrared detector material, especially in the MWIR, and is commonly cooled to suppress thermally generated carriers.","ir_review"),
("ingaas_identity","What is InGaAs?",
 "InGaAs is indium gallium arsenide, InₓGa₁₋ₓAs, a ternary III–V semiconductor. The common InP-lattice-matched composition is near In₀.₅₃Ga₀.₄₇As and is widely used for SWIR/NIR photodiodes, typically with response extending to about 1.7 µm. Extended-InGaAs compositions can reach longer wavelengths with increased dark-current challenges.","ingaas"),
("inas_identity","What is InAs?",
 "InAs is indium arsenide, a binary III–V narrow-gap semiconductor used in infrared optoelectronics and as a constituent of alloys and superlattices such as InAsSb and InAs/GaSb type-II structures.","ir_review"),
("inas_sb","What is InAsSb?",
 "InAsSb is indium arsenide antimonide, a ternary III–V alloy whose bandgap can be tuned by arsenic/antimony composition. It is used in mid- and long-wave infrared detector research, including barrier-device architectures.","ir_review"),
("t2sl","What is a type-II superlattice infrared detector?",
 "A type-II superlattice (T2SL) uses alternating semiconductor layers, commonly InAs/GaSb or related III–V systems, whose band alignment spatially separates electron and hole states. Layer thickness and composition engineer the effective gap and transport for infrared detection.","ir_review"),
("qwig","What is a QWIP?",
 "A QWIP is a quantum-well infrared photodetector. It uses intersubband absorption in semiconductor quantum wells, often GaAs/AlGaAs, rather than bulk interband absorption. Optical coupling and polarization-selection rules are central design constraints.","ir_review"),
("hgte","What is HgTe in infrared optoelectronics?",
 "HgTe is mercury telluride, a II–VI semimetal/narrow-gap constituent of HgCdTe. HgTe is also used in quantum wells and colloidal quantum dots for infrared optoelectronics. It is not interchangeable with HgCdTe because Cd fraction changes the electronic structure.","ir_review"),
("pbse","What is PbSe used for in infrared detection?",
 "PbSe is lead selenide, a lead-salt semiconductor historically used for mid-infrared photoconductive detection, often near room temperature or with modest cooling depending on the device. It is distinct from HgCdTe and InSb.","ir_review"),
("pbs","What is PbS used for in infrared detection?",
 "PbS is lead sulfide, a lead-salt narrow-gap semiconductor used in near- and short-wave infrared photoconductive detectors and in colloidal quantum-dot research.","ir_review"),
("ge_ir","What role does germanium play in infrared photonics?",
 "Germanium is a group-IV semiconductor transparent over parts of the infrared and used in IR optics, photonics and some detector/heteroepitaxial contexts. Its role depends strongly on wavelength; it is not a universal infrared detector material.","ir_review"),
("si_ir","Can silicon detect infrared light?",
 "Silicon photodiodes efficiently detect visible and near-infrared photons only up to the silicon band-edge region near 1.1 µm. Longer-wavelength SWIR/MWIR/LWIR detection generally requires other materials or engineered structures.","ir_review"),
("photodiode","What is an infrared photodiode?",
 "An infrared photodiode is a junction photon detector in which absorbed photons generate carriers that are separated and collected by a built-in or applied electric field. Responsivity, dark current, capacitance, quantum efficiency and bandwidth depend on material, junction design, area, bias and temperature.","ir_review"),
("photoconductor","What is an infrared photoconductor?",
 "An infrared photoconductor is a biased semiconductor whose conductivity changes under illumination. It can exhibit photoconductive gain when carrier lifetime exceeds transit time, but bias also introduces Joule heating, contact effects and excess-noise tradeoffs.","ir_review"),
("bolometer","What is a bolometer?",
 "A bolometer is a thermal detector: absorbed radiation changes the detector temperature and therefore a temperature-dependent electrical property, commonly resistance. Unlike a photon detector, its primary transduction mechanism is thermal rather than direct interband photoexcitation.","ir_review"),
("thermopile","What is a thermopile infrared detector?",
 "A thermopile is a thermal infrared detector built from multiple thermocouples. Absorbed radiation creates a temperature difference that generates a thermoelectric voltage. It is broadband but generally slower than many photon detectors.","ir_review"),
("pyroelectric","What is a pyroelectric infrared detector?",
 "A pyroelectric detector uses a material whose spontaneous polarization changes with temperature. It responds to changes in incident radiant power, so modulation or motion is normally required rather than a perfectly steady DC scene.","ir_review"),
("photon_vs_thermal","How are infrared photon detectors different from thermal detectors?",
 "Photon detectors convert absorbed photons directly into electronic excitations and usually have a wavelength cutoff tied to electronic energy levels. Thermal detectors respond to heating and are typically broadband; their speed and sensitivity are governed by heat capacity, thermal conductance and readout noise.","ir_review"),
("detector_types","List major types of infrared photodetectors.",
 "Major infrared detector classes include photovoltaic photodiodes, photoconductors, avalanche photodiodes, quantum-well infrared photodetectors (QWIPs), quantum-dot detectors, type-II-superlattice/barrier detectors, and thermal detectors such as bolometers, thermopiles and pyroelectrics. The best taxonomy first separates photon detectors from thermal detectors.","ir_review"),
("ir_materials","List important infrared detector material systems.",
 "Important IR detector material systems include HgCdTe (MCT), InSb, InGaAs, InAs/InAsSb, InAs/GaSb and related type-II superlattices, GaAs/AlGaAs QWIPs, lead salts such as PbS/PbSe, HgTe quantum structures/quantum dots, and thermal-detector materials such as VOx or amorphous silicon. Which is appropriate depends on wavelength, temperature, speed, noise, manufacturability and cost.","ir_review"),
("swir_materials","What materials are commonly used for SWIR detection?",
 "Common SWIR detector technologies include InGaAs photodiodes, extended-InGaAs for longer SWIR, HgCdTe with suitable composition, and some quantum-dot/lead-salt approaches. Silicon covers only the short-wavelength edge of the near-IR and does not span the usual full SWIR band.","ingaas"),
("mwir_materials","What materials are commonly used for MWIR detection?",
 "MWIR photon detectors commonly use InSb, HgCdTe, InAsSb/barrier structures and III–V type-II superlattices. Thermal detectors can also sense MWIR but operate by heating rather than a semiconductor band-edge response.","ir_review"),
("lwir_materials","What materials are commonly used for LWIR detection?",
 "LWIR photon-detector technologies include HgCdTe, III–V type-II superlattices and QWIPs; thermal microbolometers such as VOx or amorphous-silicon devices are also widely used for LWIR imaging. Cooling requirements and performance differ strongly among these technologies.","ir_review"),
("mbe","What is MBE?",
 "MBE means molecular beam epitaxy. In high vacuum, controlled atomic or molecular beams impinge on a heated crystalline substrate to grow epitaxial layers. MBE is valuable for composition, doping and multilayer control but requires careful flux, temperature and interface management.","czt_hgcdte"),
("mocvd","What is MOCVD?",
 "MOCVD means metal-organic chemical vapor deposition, also called MOVPE in many contexts. Volatile precursors react or decompose near a heated substrate to grow epitaxial semiconductor layers; precursor chemistry, flow, pressure and temperature control incorporation and defects.","ir_review"),
("lpe","What is LPE?",
 "LPE means liquid-phase epitaxy. A crystalline epilayer grows from a supersaturated liquid solution onto a crystalline substrate. LPE has long been used for HgCdTe on CdZnTe and can produce high-quality material, although composition control, melt thermodynamics and segregation must be managed.","czt_hgcdte"),
("substrates_hgcdte","What substrates are used for HgCdTe epitaxial growth?",
 "CdZnTe is the conventional near-lattice-matched HgCdTe substrate. Alternative platforms include Si, GaAs and Ge, typically with buffer layers and substantially more lattice/thermal-mismatch engineering. The growth method and buffer design must be stated before comparing them.","czt_hgcdte"),
("passivation","What does passivation mean for an infrared semiconductor detector?",
 "Passivation is treatment or coating of an exposed semiconductor surface/interface to reduce chemically or electrically active surface states, leakage, recombination and instability. It must be judged electrically and optically; a film called 'passivation' is not automatically beneficial.","ir_review"),
("anneal","Why anneal HgCdTe?",
 "HgCdTe annealing can change point-defect populations, compensation, dopant activation, carrier concentration, lifetime and surface chemistry. Mercury chemical potential/overpressure, temperature and time matter. Electrical behavior can change even when optical cutoff changes little.","ir_review"),
("etch","Why is HgCdTe etching process-dependent?",
 "HgCdTe wet or dry etching changes both geometry and surface chemistry. The best process depends on whether the goal is mesa definition, damage removal, oxide preparation or contact formation; etch residue, Te-rich surfaces, roughness and plasma damage can affect leakage and recombination.","ir_review"),
("contacts","Why do electrical contacts matter in HgCdTe measurements?",
 "Contact resistance, barriers, injection and non-ohmic behavior can distort measured detector resistance, I–V curves, responsivity and noise. Four-terminal or geometry-aware measurements and bias/temperature sweeps help separate bulk transport from contact effects.","ir_review"),
("ftir","What is FTIR spectroscopy used for in infrared detector work?",
 "FTIR spectroscopy measures an interferogram versus optical path difference and Fourier-transforms it to obtain a spectrum. It is used to characterize transmission, absorption, spectral response, filters and detector cutoff; resolution and apodization conventions must be reported.","ir_review"),
("hall","What does a Hall measurement tell you about a semiconductor?",
 "Under a valid single-carrier approximation, Hall measurements provide carrier sign/type, Hall coefficient, carrier concentration and Hall mobility when combined with conductivity. Multicarrier transport, Hall factor, contacts, geometry and parallel conduction can invalidate the simplest formulas.","ir_review"),
("blackbody","Why use a blackbody source to test infrared detectors?",
 "A calibrated blackbody provides predictable temperature-dependent spectral radiance described by Planck's law. Detector calibration still requires source emissivity, aperture/étendue, optical transmission, detector spectral response and modulation convention.","ir_review"),
("lockin","Why use a lock-in amplifier in infrared detector measurements?",
 "A lock-in amplifier extracts the component of a detector signal synchronous with a reference modulation such as a chopper. It rejects much out-of-band noise, but its amplitude convention, time constant, filter order and equivalent noise bandwidth must be included in quantitative interpretation.","ir_review"),
("nasd","What does NASD mean in FIRM detector measurements?",
 "In FIRM's detector-measurement context, NASD means noise amplitude spectral density: the square root of a noise power spectral density. Voltage NASD has units V/√Hz and current NASD has units A/√Hz. It is a spectral density, not band-integrated RMS noise.","ir_review"),
("psd_asd","How do PSD and ASD differ?",
 "Power spectral density has squared-amplitude units per hertz, such as V²/Hz. Amplitude spectral density is its square root, such as V/√Hz. RMS noise over a band is obtained by integrating the PSD through the actual transfer function and taking a square root.","ir_review"),
("nep","What is NEP?",
 "Noise-equivalent power is the input optical power that would produce signal-to-noise ratio 1 under a specified noise/bandwidth convention. Spectral NEP has units W/√Hz and equals output noise ASD divided by matching responsivity. Integrated RMS NEP has units W and depends on bandwidth.","ir_review"),
("dstar","What is D* and how is it calculated?",
 "Specific detectivity D* normalizes detector sensitivity for active area and noise bandwidth. When NEP is a spectral density in W/√Hz, D* = √(A_cm²)/NEP_density and is reported in Jones = cm·√Hz/W. Do not insert wavelength or an arbitrary bandwidth multiplier into this spectral-density form.","ir_review"),
("responsivity","What is detector responsivity?",
 "Responsivity is output signal divided by a consistently defined incident optical power. Current responsivity is A/W and voltage responsivity is V/W. It depends on wavelength, bias, gain, collection efficiency, optical coupling, temperature and modulation frequency.","ir_review"),
("qe","How is quantum efficiency related to photodiode responsivity?",
 "For an ideal unity-gain photodiode, current responsivity is R_i = ηqλ/(hc), so η = R_i hc/(qλ). Internal gain or multiplication must be separated from quantum efficiency.","ir_review"),
("johnson","What is Johnson noise?",
 "In the classical equilibrium limit, a resistor R at temperature T has one-sided open-circuit voltage-noise PSD 4k_BTR V²/Hz and ASD √(4k_BTR) V/√Hz. Band-integrated RMS noise additionally depends on the measurement transfer function or ENBW.","ir_review"),
("shot","What is shot noise?",
 "For ideal Poissonian current flow with average current I, one-sided shot-noise current ASD is √(2q|I|) A/√Hz. Excess multiplication, generation–recombination processes or electronics can make measured noise larger.","ir_review"),
("gr_noise","What is generation-recombination noise?",
 "Generation–recombination noise arises from stochastic fluctuations in carrier number as carriers are generated and recombine. A single characteristic time can produce a Lorentzian spectrum, but real devices may contain multiple lifetimes or traps; a fitted corner alone does not identify a microscopic defect.","ir_review"),
("flicker","What is 1/f noise?",
 "1/f or flicker noise is excess low-frequency noise whose PSD often scales approximately as 1/f^α over some frequency range. Its magnitude can depend on bias, contacts, surfaces and material quality. One frequency point cannot establish a trap mechanism.","ir_review"),
("time_constant","What is a detector time constant?",
 "A detector time constant describes a characteristic response timescale. For a true isolated single-pole response, f_3dB = 1/(2πτ). Real detectors can have multiple poles from carrier dynamics, RC loading, traps, thermal response and readout electronics.","ir_review"),
("db","What is the formula for decibels?",
 "For a power ratio, dB = 10 log10(P2/P1). For an amplitude ratio measured at equal impedance, dB = 20 log10(|A2/A1|). Always state the reference quantity; using 20 log for power or 10 log for voltage amplitude is incorrect.","ir_review"),
("cutoff","How is detector cutoff wavelength related to bandgap?",
 "A first-order band-edge estimate is E_g ≈ hc/λ_c, or E_g[eV] ≈ 1.23984/λ_c[µm]. A measured detector cutoff also depends on the response criterion, temperature, absorption tails, device optics and composition gradients.","ir_review"),
("uncertainty","How should FIRM handle an underspecified detector question?",
 "State what information is missing, give a conditional formula or physically reasonable range when useful, and separate assumptions from measured facts. Do not invent temperature, geometry, bias, bandwidth, material composition or a microscopic mechanism merely to force a numerical answer.","ir_review"),
("qcl","What is a quantum cascade laser?",
 "A quantum cascade laser (QCL) is a unipolar semiconductor laser that uses engineered intersubband transitions in repeated quantum-well stages. Its emission wavelength is set mainly by layer design rather than a bulk bandgap, enabling mid-IR and THz sources for spectroscopy and sensing.","qcl_review"),
("icl","What is an interband cascade laser?",
 "An interband cascade laser (ICL) is a mid-infrared semiconductor laser that combines interband electron-hole recombination, commonly in type-II structures, with a cascaded active-region architecture. ICLs can offer lower electrical power than many QCL implementations in parts of the mid-IR.","mid_ir_sources"),
("qcl_vs_icl","How do QCLs and ICLs differ?",
 "QCLs are unipolar devices based on intersubband transitions in repeated quantum-well stages, whereas ICLs use interband electron-hole recombination in a cascaded type-II architecture. Both are important mid-IR coherent sources, but their carrier physics, voltage/current requirements and useful spectral ranges differ.","mid_ir_sources"),
("mir_fingerprint","Why is the mid-infrared important for chemical sensing?",
 "Many molecules have strong vibrational absorption features in the mid-infrared, so MIR spectroscopy can provide chemically specific spectral fingerprints. Source linewidth/tunability, optical path, detector sensitivity and gas/sample absorption strength determine system performance.","mid_ir_platform"),
("chalcogenide_glass","What is a chalcogenide glass in infrared photonics?",
 "Chalcogenide glasses are amorphous materials containing one or more chalcogen elements such as sulfur, selenium or tellurium, often with elements such as As, Ge or Sb. Many compositions have broad mid-IR transparency, high refractive index and useful nonlinear optical properties for fibers, waveguides and sensing.","chalcogenide"),
("mir_waveguide","What materials are used for mid-infrared waveguides?",
 "Mid-IR waveguide platforms include silicon or germanium in suitable wavelength ranges, chalcogenide glasses, and other specialty semiconductors/dielectrics. Material transparency, substrate absorption, confinement, nonlinear loss, fabrication roughness and wavelength determine the practical choice.","mid_ir_platform"),
("evanescent_sensing","How does an infrared waveguide perform evanescent-field sensing?",
 "A guided optical mode extends partly outside the core as an evanescent field. Molecules near that field absorb at their characteristic infrared wavelengths, changing transmitted intensity or phase. Interaction length, modal overlap, analyte concentration and propagation loss determine sensitivity.","mid_ir_platform"),
("microbolometer","What is a microbolometer?",
 "A microbolometer is an uncooled thermal infrared detector pixel whose absorber/thermistor is thermally isolated from the substrate. Incident radiation changes pixel temperature and resistance; sensitivity and speed are governed by thermistor coefficient, thermal conductance, heat capacity, readout noise and pixel geometry.","microbolometer"),
("vox","What is VOx in infrared imaging?",
 "VOx denotes vanadium-oxide-based thermistor materials widely used in uncooled microbolometer focal-plane arrays. Their temperature-dependent resistance enables thermal infrared detection; process uniformity, temperature coefficient of resistance and 1/f/readout noise affect array performance.","microbolometer"),
("amorphous_si_bolo","How is amorphous silicon used in infrared detectors?",
 "Hydrogenated amorphous silicon and related silicon films can serve as thermistor materials in uncooled microbolometers. They are thermal detectors rather than band-edge LWIR photon detectors; CMOS-process compatibility is one reason they are attractive.","microbolometer"),
("roic","What is a ROIC in an infrared focal-plane array?",
 "A ROIC is the readout integrated circuit coupled to detector pixels. It provides functions such as biasing, charge/current integration, gain, multiplexing, timing and often digitization. Read noise, well capacity, linearity, capacitance and bandwidth can limit an otherwise excellent detector material.","roic_review"),
("fpa","What is an infrared focal-plane array?",
 "An infrared focal-plane array (IRFPA) is a two-dimensional pixel array that converts an infrared image at the focal plane into electrical signals. Detector material, pixel architecture, ROIC, hybridization or monolithic integration, optics, cooling and calibration all contribute to final imaging performance.","roic_review"),
("netd","What does NETD mean for a thermal imager?",
 "NETD is noise-equivalent temperature difference: the scene-temperature change that produces signal comparable to system noise under specified optics, spectral band, integration time and processing. It is a system-level imager metric and is not interchangeable with detector D*.","microbolometer"),
("blip","What does background-limited infrared photodetection mean?",
 "Background-limited infrared photodetection (BLIP) means photon fluctuations from the optical background dominate other detector and readout noise under the stated operating conditions. BLIP depends on temperature, field of view, spectral band, optics and detector noise; it is not an intrinsic material label.","hot_review"),
("r0a","What is R0A for an infrared photodiode?",
 "R0A is zero-bias differential resistance multiplied by detector area, commonly used to compare photodiode dark-current/leakage behavior while reducing simple area scaling. Its interpretation depends on temperature, junction design and whether surface or bulk currents dominate.","hot_review"),
("dark_current","What is dark current in an infrared photodiode?",
 "Dark current is electrical current measured without the intended optical signal. In narrow-gap IR photodiodes it can contain diffusion, generation-recombination, tunneling, surface leakage and contact-related components; dominant mechanisms depend on temperature, bias, material and device architecture.","hot_review"),
("srh","What is Shockley-Read-Hall recombination?",
 "Shockley-Read-Hall (SRH) recombination is trap-assisted electron-hole recombination through defect levels in the bandgap. Its rate depends on trap energy, density, capture cross sections and carrier concentrations, and it can strongly influence lifetime and generation current in infrared semiconductors.","hot_review"),
("auger","What is Auger recombination in narrow-gap infrared semiconductors?",
 "Auger recombination is a nonradiative three-carrier process in which electron-hole recombination transfers energy to another carrier. It can become important at high carrier concentration and in narrow-gap materials, contributing to lifetime and dark-current limits in some HgCdTe operating regimes.","hot_review"),
("radiative_recomb","What is radiative recombination in a semiconductor?",
 "Radiative recombination occurs when an electron and hole recombine and emit a photon. In direct-gap semiconductors it competes with nonradiative channels such as SRH and Auger processes; the dominant lifetime mechanism depends on carrier density, temperature, defects and band structure.","hot_review"),
("tunneling","Why does tunneling matter in narrow-gap infrared photodiodes?",
 "Narrow bandgaps and strong electric fields can make band-to-band or trap-assisted tunneling significant, increasing dark current especially at reverse bias. Junction doping profile, field distribution, defects and temperature help distinguish tunneling from thermal generation mechanisms.","hot_review"),
("depletion","What is the depletion region in a photodiode?",
 "The depletion region is the space-charge region near a semiconductor junction where mobile majority carriers are depleted and an electric field exists. Photocarriers generated there can be swept by the field, while depletion width and field also influence capacitance, generation current and tunneling.","ir_review"),
("diffusion_length","What is minority-carrier diffusion length?",
 "For a simple uniform semiconductor, minority-carrier diffusion length is L = √(Dτ), where D is diffusion coefficient and τ an effective minority-carrier lifetime. Surface recombination, electric fields, trapping and nonuniform material can make the effective transport length differ from this ideal model.","ir_review"),
("einstein","What is the Einstein relation for semiconductor carriers?",
 "For nondegenerate carriers in thermal equilibrium, diffusion coefficient and mobility are related by D/μ = k_BT/q. Degenerate statistics, strong fields and non-equilibrium transport can require corrections.","ir_review"),
("depletion_cap","Why does photodiode capacitance matter?",
 "Junction capacitance combines with detector/readout resistance and amplifier input capacitance to influence bandwidth and noise. Capacitance also affects charge integration and readout stability; smaller pixels can reduce capacitance but introduce other optical/electrical tradeoffs.","roic_review"),
("ar_coating","Why use an anti-reflection coating on an infrared detector?",
 "An anti-reflection coating uses thin-film interference to reduce Fresnel reflection and increase optical coupling over a designed wavelength/angle range. Material absorption, refractive index, thermal behavior and fabrication tolerance determine performance.","ir_review"),
("cold_shield","What is a cold shield in a cooled infrared detector system?",
 "A cold shield limits the detector's field of view to the intended optical aperture and blocks unwanted warm radiation from surrounding structures. It reduces background loading and stray radiation, which is especially important for sensitive cooled MWIR/LWIR detectors.","hot_review"),
("etendue","What is optical étendue and why does it matter in infrared systems?",
 "Étendue is an area-solid-angle throughput quantity conserved in passive lossless geometrical optics, with refractive-index factors where appropriate. It connects source/aperture geometry, detector area and accepted solid angle, so it matters for radiometric power and background loading.","mid_ir_platform"),
("planck","What does Planck's law describe in infrared radiometry?",
 "Planck's law gives the spectral radiance of an ideal blackbody as a function of wavelength or frequency and temperature. Real calibration requires emissivity, aperture/étendue, optical transmission and the detector's spectral response in the same spectral-density convention.","ir_review"),
("wien","What does Wien's displacement law tell you?",
 "For the wavelength-domain blackbody spectrum, Wien's law gives λ_peak T ≈ 2898 µm·K. It identifies the wavelength of peak spectral radiance per unit wavelength; the frequency-domain spectrum peaks at a different mapped wavelength because the spectral-density variable changes.","ir_review"),
("atmospheric_windows","Why do 3–5 µm and 8–12 µm matter in infrared sensing?",
 "The atmosphere has relatively transmissive windows in portions of the MWIR near 3–5 µm and LWIR near 8–12 µm, which strongly influence remote-sensing and imaging system design. Exact transmission depends on path length, humidity, gases and weather.","mid_ir_sources"),
("fresnel","What determines Fresnel reflection at an infrared optical interface?",
 "Fresnel reflection is set by refractive indices, polarization and incidence angle. High-index IR materials can have substantial surface reflection, motivating anti-reflection coatings or subwavelength structures. Absorption must be treated separately from interface reflection.","mid_ir_platform"),
("beer_lambert_fact","What is the Beer-Lambert absorption relation?",
 "For a uniform non-scattering medium with absorption coefficient α and path length d, intensity transmission is T = exp(-αd) when interface reflections are excluded. Real semiconductor films can require Fresnel, interference, scattering and substrate corrections.","ir_review"),
("spectral_density_jacobian","Why can’t a spectrum per µm be relabeled as a spectrum per cm⁻¹?",
 "Spectral densities transform with the Jacobian of the coordinate change so integrated power is conserved. A value per unit wavelength and a value per unit wavenumber therefore have different numerical magnitudes and even different apparent peak locations.","mid_ir_platform"),
]

TYPO_WORDS={"photodetectors":["photodetectors","photo detectors","photdetectors","photodetecors"],
"MCT":["MCT","mct","HgCdTe","mercury cadmium telluride"],
"CZT":["CZT","czt","CdZnTe","cadmium zinc telluride"],
"InGaAs":["InGaAs","ingaas","In Ga As"],
"responsivity":["responsivity","responsivty","response per watt"],
"substrates":["substrates","subtrates","growth substrates"]}

def perturb(q:str,variant:int)->str:
    if variant==0:return q
    if variant==1:return q.lower()
    if variant==2:return q.replace("?","")
    if variant==3:return "Please answer directly: "+q
    if variant==4:return "In infrared detector work, "+q[0].lower()+q[1:]
    if variant==5:return q.replace("photodetectors","photo detectors").replace("detector","detctor",1)
    if variant==6:return q.replace("infrared","IR").replace("Infrared","IR")
    if variant==7:return q.replace(" what "," whats ").replace("What ","Whats ",1).replace("?","?")
    return q

FACT_WRAPPERS=[
    "{q}","Answer this infrared-engineering question: {q}","Give the concise technical definition: {q}",
    "A detector engineer asks: {q}","For IR photonics, {q}","In semiconductor-detector context, {q}",
    "Explain this without unrelated meanings: {q}","State the canonical meaning and key caveat: {q}",
    "Quick technical check — {q}","Please correct any common confusion and answer: {q}",
    "For a new infrared engineer, {q}","Use the material/device meaning relevant to FIRM: {q}",
    "What should an IR specialist say if asked: {q}","Give a two-sentence expert answer: {q}",
    "Without guessing beyond the field, {q}","Answer as if documenting a detector test plan: {q}",
]

def make_fact_rows():
    rows=[]
    for i,(fid,q,a,src) in enumerate(FACTS):
        for v,wrapper in enumerate(FACT_WRAPPERS):
            prompt=wrapper.format(q=perturb(q,v%8))
            rows.append({"id":f"fact_{fid}_{v:02d}","category":"canonical_fact",
              "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt},{"role":"assistant","content":a}],
              "metadata":{"fact_id":fid,"source_url":SOURCES[src],"rights":"original_paraphrase_from_verified_claims",
                          "review_status":"provisional_research_claim_checked","variant":v}})
    return rows

TAXONOMY=[
("List major photon-detector architectures used in infrared sensing.",
 "Photon-detector architectures include photoconductors, p–n/p–i–n photodiodes, avalanche photodiodes, barrier detectors, quantum-well infrared photodetectors, quantum-dot detectors and type-II-superlattice devices. Thermal detectors such as bolometers belong to a different top-level class."),
("Name representative SWIR, MWIR and LWIR material systems.",
 "Representative SWIR materials include InGaAs and suitably composed HgCdTe; MWIR includes InSb, HgCdTe, InAsSb and III–V superlattices; LWIR includes HgCdTe, III–V type-II superlattices and QWIPs, while thermal microbolometers also image LWIR without a semiconductor band-edge cutoff."),
("Which detector metrics should be reported together when comparing infrared detectors?",
 "At minimum report wavelength/band, operating temperature, active area, bias, responsivity or quantum efficiency, noise spectral density or NEP with bandwidth convention, D* if used, response bandwidth/time constant, optical background and whether values are measured or inferred."),
("What measurements help diagnose an HgCdTe detector?",
 "Useful measurements include spectral response/transmission, I–V or resistance versus bias, noise PSD/ASD versus frequency, frequency response, Hall carrier concentration/mobility, temperature dependence, calibrated blackbody responsivity, contact tests and process/geometry controls."),
]

def make_taxonomy_rows():
    rows=[]
    for i,(q,a) in enumerate(TAXONOMY):
        for v in range(10):
            rows.append({"id":f"taxonomy_{i}_{v}","category":"taxonomy",
              "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":perturb(q,v%8)},{"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research","rights":"original"}})
    return rows

def eg_hansen(x,T):
    return -0.302+1.93*x-0.81*x*x+0.832*x**3+5.35e-4*(1-2*x)*T
def inv_hansen(lam,T):
    target=HC/lam; lo,hi=0.,1.
    for _ in range(100):
        m=(lo+hi)/2
        if eg_hansen(m,T)<target:lo=m
        else:hi=m
    return (lo+hi)/2,target

def quantitative_rows(seed=20261007):
    r=random.Random(seed); rows=[]
    # MCT composition: avoid exact stress-test 8.1um/300K pair.
    for i in range(320):
        T=r.choice([77.,100.,150.,200.,250.,300.])
        lam=r.uniform(2.5,14.5)
        if abs(lam-8.1)<0.08 and T==300.: lam+=0.23
        x,eg=inv_hansen(lam,T)
        q=(f"Using the empirical HgCdTe relation Eg(x,T) = -0.302 + 1.93x - 0.81x^2 + "
           f"0.832x^3 + 5.35e-4(1-2x)T eV, estimate Cd mole fraction x for an ideal "
           f"cutoff of {lam:.4g} µm at {T:.0f} K.")
        a=(f"First convert cutoff to gap: Eg ≈ 1.239841984/λ = {eg:.6g} eV. "
           f"Solving the stated empirical Eg(x,T) relation on 0≤x≤1 gives x ≈ {x:.5f}. "
           "This is a model-dependent band-edge composition estimate; measured cutoff criteria and composition can differ.")
        rows.append({"id":f"quant_hansen_{i:04d}","category":"quantitative_open",
          "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":q},{"role":"assistant","content":a}],
          "metadata":{"oracle":"programmatic_bisection","family":"hgcdte_hansen","review_status":"formula_verified"}})
    # D*, NEP, Johnson, responsivity, dB open prose examples
    for i in range(500):
        fam=i%5
        if fam==0:
            area=10**r.uniform(-2,1); nep=10**r.uniform(-13,-9); val=math.sqrt(area*.01)/nep
            q=f"An IR detector has active area {area:.5g} mm^2 and spectral NEP {nep:.5g} W/sqrt(Hz). Calculate D* in Jones."
            a=f"Convert area: A={area*.01:.6g} cm². For spectral NEP, D*=√A/NEP = {val:.6g} Jones (cm·√Hz/W). No wavelength or extra bandwidth factor belongs in this form."
        elif fam==1:
            noise=10**r.uniform(-10,-7); resp=10**r.uniform(2,6); val=noise/resp
            q=f"Voltage noise ASD is {noise:.5g} V/sqrt(Hz) and voltage responsivity is {resp:.5g} V/W. Find spectral NEP."
            a=f"NEP_density = e_n/R_V = {val:.6g} W/√Hz. Both quantities must refer to the same frequency and operating conditions."
        elif fam==2:
            R=10**r.uniform(1,5);T=r.uniform(60,320);val=math.sqrt(4*K*T*R)
            q=f"Find one-sided Johnson voltage ASD for R={R:.5g} ohm at T={T:.4g} K."
            a=f"e_n=√(4k_BTR)={val:.6g} V/√Hz in the classical equilibrium limit. This is ASD, not RMS noise over a finite bandwidth."
        elif fam==3:
            eta=r.uniform(.2,.95);lam=r.uniform(1,10);val=eta*lam/HC
            q=f"An ideal unity-gain photodiode has external quantum efficiency {eta:.4g} at {lam:.4g} µm. Find current responsivity."
            a=f"R_i=ηqλ/(hc)=ηλ/1.239841984 = {val:.6g} A/W, assuming one collected carrier per detected photon and no internal gain."
        else:
            ratio=10**r.uniform(-2,2); amp=(i%2==0)
            if amp:
                val=20*math.log10(ratio);q=f"An equal-impedance voltage amplitude changes by a ratio {ratio:.5g}. Express the ratio in dB."
                a=f"For an amplitude ratio at equal impedance, dB=20log10(A2/A1)={val:.6g} dB."
            else:
                val=10*math.log10(ratio);q=f"A power changes by a ratio {ratio:.5g}. Express the ratio in dB."
                a=f"For a power ratio, dB=10log10(P2/P1)={val:.6g} dB."
        rows.append({"id":f"quant_open_{i:04d}","category":"quantitative_open",
          "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":q},{"role":"assistant","content":a}],
          "metadata":{"oracle":"programmatic","review_status":"formula_verified"}})
    # Natural-language versions of the full verified quantitative curriculum.
    # This teaches conversational use of formulas rather than only strict JSON output.
    try:
        from build_firm_quantitative_sft import FAMILIES as BASE_FAMILIES, case as base_case
    except Exception:
        BASE_FAMILIES=[]; base_case=None
    for fam in BASE_FAMILIES:
        for j in range(70):
            rr=random.Random(f"v2-open-{seed}-{fam}-{j}")
            prompt,val,unit,key,explanation,checker=base_case(fam,rr)
            if not checker(val): raise AssertionError((fam,j,val))
            answer=f"{explanation} Final result: {val:.7g} {unit}."
            rows.append({"id":f"quant_family_{fam}_{j:03d}","category":"quantitative_open",
              "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt},{"role":"assistant","content":answer}],
              "metadata":{"oracle":"existing_formula_verified_generator","family":fam,"review_status":"formula_verified"}})
    return rows

OUTSIDE=[
"Write a chocolate chip cookie recipe.","Who is a cartoon character named Woody Woodpecker?",
"Give me stock picks for next year.","Plan a vacation itinerary for Paris.","Write a poem about a birthday.",
"Who should I vote for?","Recommend a television series.","Help me choose a football fantasy lineup.",
"Design my living room.","Teach me guitar chords.","Tell me celebrity gossip.","Recommend a restaurant."
]
REFUSAL="FIRM specializes in infrared photonics and related semiconductor science. That request is outside my scope."

def refusal_rows():
    rows=[]
    for i,q in enumerate(OUTSIDE):
        for v in range(3):
            rows.append({"id":f"outside_{i}_{v}","category":"out_of_scope",
              "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":perturb(q,v)},{"role":"assistant","content":REFUSAL}],
              "metadata":{"review_status":"scope_checked","rights":"original"}})
    return rows

def diagnostic_rows():
    base=[
("A detector's low-frequency noise rises after passivation. Does that prove the passivation created traps?",
 "No. It is consistent with a process-induced change but does not uniquely identify traps. Compare pre/post bias dependence, contact behavior, resistance, temperature scaling, PSD shape, surface chemistry and a dummy/readout baseline before assigning a microscopic mechanism."),
("An HgCdTe detector has a -3 dB point at 20 kHz. Is carrier lifetime definitely 1/(2πf)?",
 "Only if the measured response is truly dominated by a single detector pole. Chopper, preamplifier, RC loading, lock-in filters, diffusion and trapping can add poles. De-embed the measurement chain before interpreting τ=1/(2πf3dB) as carrier lifetime."),
("A device has high responsivity but also high noise. Is it a good detector?",
 "Not necessarily. Sensitivity depends on signal relative to noise. Compare responsivity with noise spectral density or NEP under the same conditions and then D* if area normalization is appropriate; high gain can increase both responsivity and noise."),
("A dark I-V curve looks ideal. Does that establish good infrared performance?",
 "No. Dark I-V constrains electrical leakage/conduction but does not establish spectral absorption, carrier collection, responsivity, bandwidth or optical alignment. Combine I-V with spectral response, calibrated optical signal and noise measurements."),
("Passivation lowers noise ASD but also lowers responsivity. Did detector performance improve?",
 "Compare NEP or D* under identical conditions. If noise ASD falls by a larger fractional amount than responsivity, NEP can improve despite lower signal. Then investigate whether the responsivity loss is optical, contact-related, gain-related or recombination-related using spectral, resistance and frequency-response controls."),
("A fitted Lorentzian noise corner corresponds to 8 microseconds. Does that identify the minority-carrier lifetime?",
 "No. A Lorentzian correlation time may reflect a trap, generation-recombination process, contact fluctuation or circuit pole. Compare it with optical modulation response, transients, temperature dependence and readout transfer functions before identifying a carrier lifetime."),
("A detector's resistance changed strongly after annealing but its optical cutoff barely moved. Is that contradictory?",
 "No. Cutoff mainly tracks bandgap/composition, while annealing can alter point defects, compensation, dopant activation, carrier concentration, mobility, lifetime and contacts. Electrical properties can change substantially without a comparable band-edge shift."),
("How can Hall measurements constrain an HgCdTe photoconductor model?",
 "Hall carrier concentration and mobility constrain conductivity and expected bulk resistance, and mobility constrains drift transit time. Large disagreement with device resistance can indicate geometry error, contact resistance, surface or parallel conduction, compensation or nonuniform active thickness. Hall data does not directly supply lifetime or trap spectra."),
("How do you determine whether a detector is Johnson-noise-limited or readout-limited?",
 "Predict the Johnson ASD from measured R and T and independently measure the readout with an equivalent dummy impedance. Then compare spectral shape and scaling versus resistance, temperature and bias. A flat measured floor alone cannot distinguish detector thermal noise from amplifier noise."),
("How do you test whether an infrared detector is background-limited?",
 "Vary controlled optical background, aperture, cold shield or blackbody temperature and test whether signal/noise scale with photon statistics while readout and dark contributions remain subdominant. BLIP is an operating-condition claim, not a material label."),
("Why can changing chopper frequency change measured responsivity?",
 "Detector carrier/thermal dynamics, preamplifier poles, RC loading and lock-in filters all have frequency response. Sweep modulation frequency and de-embed the electronics before attributing rolloff to the detector alone."),
("An HgCdTe transmission spectrum slopes downward at wavelengths beyond the band edge. Is that automatically interband absorption?",
 "No. Fresnel effects, free-carrier absorption, substrate transmission, scattering, interference and baseline drift can shape the long-wavelength region. Use substrate/background correction and thickness/process comparisons before extracting absorption physics."),
("How should a blackbody measurement be converted into detector responsivity?",
 "Integrate blackbody spectral radiance over wavelength, source aperture/solid angle, detector active area, optical transmission, emissivity and detector spectral response to obtain modulated power at the detector. Then divide the correctly referenced electrical signal by that modulated optical power using consistent RMS/peak conventions."),
("A detector has a large signal only at high bias. What should be checked before calling it high gain?",
 "Check self-heating, contact injection, non-ohmic behavior, resistance change, noise growth, saturation and readout limits. Photoconductive gain can rise with bias through shorter transit time, but a rising output alone does not prove useful gain."),
("Two detectors have the same D* but different active areas and bandwidths. Are they equivalent?",
 "No. D* is a normalized sensitivity metric, not a complete system specification. Compare spectral band, temperature, background, speed, saturation, area, optics, noise spectrum and whether D* was measured or inferred."),
("A measured noise spectrum has a 1/f-like slope. Does that prove surface traps dominate?",
 "No. Contacts, bulk mobility fluctuations, surfaces, readout drift and multiple Lorentzian processes can approximate a 1/f-like spectrum. Use bias, temperature, geometry, passivation and dummy-readout controls to discriminate mechanisms."),
("Why might a photoconductor with longer lifetime have worse bandwidth?",
 "Longer carrier lifetime can increase photoconductive gain but also slows carrier-number response. In the simplest lifetime-limited single-pole picture, bandwidth scales roughly as 1/(2πτ), producing a gain-bandwidth tradeoff; other poles can also contribute."),
("How can contact resistance be separated from bulk detector resistance?",
 "Use four-terminal measurements where geometry permits, compare multiple contact spacings, inspect I-V linearity and temperature dependence, and use transmission-line or dedicated contact structures. Two-terminal resistance alone mixes contacts and active material."),
("What evidence is needed before claiming an etch damaged HgCdTe?",
 "Compare matched pre/post or process-split devices using surface morphology/chemistry, leakage or resistance, spectral response, noise PSD, lifetime/frequency response and passivation recovery. One degraded electrical metric is not enough to identify etch damage uniquely."),
("How should FIRM answer when a user asks for a detector mechanism but gives only one data point?",
 "State that the mechanism is underdetermined, list the leading physically plausible hypotheses, and specify the measurements that would discriminate them. Give conditional equations if useful, but do not invent missing bias, temperature, geometry or material parameters."),
]
    rows=[]
    for i,(q,a) in enumerate(base):
        for v in range(20):
            rows.append({"id":f"diag_{i}_{v}","category":"diagnostic_reasoning",
             "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":perturb(q,v%8)},{"role":"assistant","content":a}],
             "metadata":{"review_status":"provisional_research","rights":"original"}})
    return rows

def build(out:Path):
    if out.exists(): raise FileExistsError(out)
    rawrows=make_fact_rows()+make_taxonomy_rows()+quantitative_rows()+diagnostic_rows()+refusal_rows()
    # Exact prompt duplicates can arise when a typo/paraphrase transform is a no-op.
    # Keep only the first deterministic occurrence so repetition does not overweight a fact.
    by_prompt={}
    for row in rawrows:
        prompt=row["messages"][1]["content"]
        by_prompt.setdefault(prompt,row)
    allrows=list(by_prompt.values())
    rng=random.Random(991744)
    rng.shuffle(allrows)
    nvalid=max(200,int(len(allrows)*.08))
    # stratified by category
    cats={}
    for x in allrows: cats.setdefault(x["category"],[]).append(x)
    train=[];valid=[]
    for cat,rows in cats.items():
        k=max(2,round(len(rows)*.08))
        valid+=rows[:k]; train+=rows[k:]
    rng.shuffle(train);rng.shuffle(valid)
    out.mkdir(parents=True)
    for split,rows in [("train",train),("valid",valid)]:
        p=out/(split+".jsonl")
        p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows))
    manifest={"schema_version":"2.0","rows":{"train":len(train),"valid":len(valid)},
      "categories":{cat:sum(1 for r in allrows if r["category"]==cat) for cat in sorted(cats)},
      "foundational_fact_cards":len(FACTS),"source_urls":sorted(set(SOURCES.values())),
      "private_benchmark_imported":False,"exact_user_stress_prompts_imported":False,
      "status":"PROVISIONAL_RESEARCH_CLAIM_CHECKED_NOT_EXPERT_RELEASE_APPROVED",
      "train_sha256":hashlib.sha256((out/"train.jsonl").read_bytes()).hexdigest(),
      "valid_sha256":hashlib.sha256((out/"valid.jsonl").read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("data/processed/firm4b_v2_foundation_v1"));a=ap.parse_args();build(a.out)
