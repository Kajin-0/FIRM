#!/usr/bin/env python3
"""Expanded FIRM-4B v2.1 curriculum.

Builds on v2 foundation but increases *unique conceptual targets*, not merely
prompt wrappers. No private benchmark prompts/gold are read.
"""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import build_firm4b_v2_foundation as base

EXTRA_SOURCES={
"inassb_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7763214/",
"t2sl_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7692601/",
"lead_cqd":"https://pmc.ncbi.nlm.nih.gov/articles/PMC10488450/",
"czt_radiation":"https://pmc.ncbi.nlm.nih.gov/articles/PMC3297127/",
"ir_history_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC3673126/",
}

# (id, question, answer, evidence key)
EXTRA_FACTS=[
("cdte","What is CdTe in infrared and detector technology?",
 "CdTe is cadmium telluride, a binary II–VI semiconductor. It is used as a substrate/buffer constituent in HgCdTe technology and separately as a room-temperature X-ray/gamma-ray detector material. Those applications should not be confused with HgCdTe infrared absorption.","czt_radiation"),
("znte","What is ZnTe and why can it matter near HgCdTe technology?",
 "ZnTe is zinc telluride, a wide-gap II–VI semiconductor. In CdZnTe, ZnTe alloying adjusts lattice parameter and bandgap; ZnTe itself is not an LWIR detector substitute for HgCdTe.","czt_radiation"),
("gasb","What is GaSb in infrared semiconductor technology?",
 "GaSb is gallium antimonide, a III–V semiconductor used as a substrate and as a constituent of antimonide heterostructures. It is especially important in InAs/GaSb type-II superlattices and mid-infrared optoelectronics.","t2sl_review"),
("inp","What role does InP play in infrared photonics?",
 "InP is indium phosphide, a III–V semiconductor widely used as a substrate/platform for near-IR and SWIR optoelectronics, including lattice-matched InGaAs photodiodes. It is not the conventional lattice-matched substrate for HgCdTe.","ir_history_review"),
("gaas","What role does GaAs play in infrared photonics?",
 "GaAs is gallium arsenide, a III–V semiconductor used in optoelectronics, quantum wells and as a foreign substrate platform with buffer layers. Bare GaAs is strongly lattice-mismatched to HgCdTe, so direct equivalence to CdZnTe is incorrect.","ir_review"),
("hgcdte_x_trend","How does increasing Cd fraction x change Hg1-xCdxTe?",
 "Increasing Cd fraction generally raises the HgCdTe bandgap and shifts the ideal absorption/cutoff edge to shorter wavelength at fixed temperature. Quantitative conversion requires a specified empirical bandgap relation and temperature.","ir_review"),
("hgcdte_temperature_gap","Why must temperature be specified when converting HgCdTe cutoff to composition?",
 "HgCdTe bandgap is temperature dependent, so the same Cd fraction has different bandgap/cutoff at different temperatures. Composition inference from cutoff is therefore conditional on temperature, the bandgap model and the cutoff criterion.","ir_review"),
("cutoff_criterion","Why does the definition of cutoff wavelength matter?",
 "A measured detector cutoff may be defined at 50% response, a fixed responsivity threshold, an extrapolated absorption edge or another convention. Different criteria can shift the reported cutoff, so composition inference should state the criterion when available.","ir_review"),
("absorption_vs_qe","Is optical absorption the same as external quantum efficiency?",
 "No. Absorption counts incident photons removed from the optical beam; external quantum efficiency counts collected charge carriers per incident photon. Reflection, parasitic absorption and recombination can make high absorption coexist with lower external QE.","ir_review"),
("qe_vs_responsivity","Is responsivity the same as quantum efficiency?",
 "No. Quantum efficiency is a carrier-per-incident-photon ratio; current responsivity is electrical current per optical power in A/W. For an ideal unity-gain photodiode they are related by R=eta*q*lambda/(hc), but gain and wavelength matter.","ir_review"),
("gain_vs_qe","Can external responsivity exceed the unity-gain quantum-efficiency limit?",
 "Yes, if the detector has internal multiplication or photoconductive gain. Responsivity above q*lambda/(hc) does not imply quantum efficiency above 100% unless gain is separately accounted for.","ir_review"),
("photoconductive_gain","What is photoconductive gain?",
 "In an idealized photoconductor, gain can arise when carrier lifetime exceeds transit time, so one photogenerated carrier pair influences current for longer than one transit. A common estimate is G≈tau_life/tau_transit, subject to contacts, trapping, field and geometry.","ir_review"),
("nbn","What is an nBn infrared detector?",
 "An nBn is a unipolar-barrier detector with n-type absorber/contact regions separated by a barrier designed to block majority-carrier dark current while allowing minority photocarriers to pass. Barrier design can suppress depletion-region generation-recombination current.","inassb_review"),
("pbn","What is a pBn or related unipolar-barrier detector?",
 "p-type variants of unipolar-barrier designs use band-engineered barriers to suppress majority-carrier dark current while transmitting the desired photocarriers. Exact layer polarity and band offsets must be specified rather than inferred from the generic barrier label.","inassb_review"),
("apd","What is an avalanche photodiode in infrared detection?",
 "An avalanche photodiode uses impact ionization to provide internal current multiplication under high electric field. Gain improves signal amplitude but introduces excess noise, bias sensitivity and breakdown constraints; multiplication physics is material dependent.","ir_review"),
("hgapd","Why are HgCdTe electron-initiated APDs unusual?",
 "HgCdTe can exhibit strongly asymmetric electron and hole impact-ionization behavior in suitable structures, enabling low-excess-noise electron-initiated avalanche gain compared with many conventional APDs. Performance still depends on composition, temperature and field profile.","ir_review"),
("qdot_detector","What is a quantum-dot infrared photodetector?",
 "A quantum-dot infrared photodetector uses discrete confined states in semiconductor quantum dots to absorb infrared photons. Quantum confinement can alter spectral selection rules and dark-current behavior compared with bulk or quantum-well detectors.","ir_history_review"),
("cqd_detector","What is a colloidal quantum-dot infrared detector?",
 "A colloidal quantum-dot detector uses solution-processed semiconductor nanocrystals, such as lead- or mercury-chalcogenide quantum dots, whose size/composition tune absorption. Surface chemistry, ligand exchange and transport between dots are central to performance.","lead_cqd"),
("pbs_cqd","Why are PbS colloidal quantum dots useful in infrared photodetection?",
 "PbS quantum dots provide size-tunable near-IR/SWIR absorption and solution-processability. Device performance depends strongly on surface passivation, ligand chemistry, film transport and dark current rather than bandgap alone.","lead_cqd"),
("pbse_cqd","Why are PbSe colloidal quantum dots used for infrared detection?",
 "PbSe quantum dots offer tunable infrared absorption and solution processing. As with other colloidal quantum-dot detectors, surface traps, ligand exchange, film transport and environmental stability strongly affect noise and responsivity.","lead_cqd"),
("t2sl_inas_gasb","What is an InAs/GaSb type-II superlattice?",
 "An InAs/GaSb T2SL alternates thin InAs and GaSb layers with type-II band alignment. Electron and hole states reside primarily in different layers, and layer thickness/composition engineer the effective infrared bandgap.","t2sl_review"),
("t2sl_inas_inassb","What is an InAs/InAsSb type-II superlattice?",
 "An InAs/InAsSb strained-layer superlattice is a gallium-free III–V infrared absorber whose effective bandgap is engineered by layer thickness and Sb content. It is actively used in MWIR and extended toward longer wavelengths.","t2sl_review"),
("qwig_selection","Why do QWIPs need special optical coupling?",
 "Intersubband transitions in conventional quantum wells obey polarization selection rules, so normally incident light may couple weakly. Gratings, mesas or other coupling structures are often used to provide an electric-field component along the growth direction.","ir_history_review"),
("photovoltaic_mode","What does photovoltaic operation mean for a photodiode?",
 "Photovoltaic operation generally means detecting the junction-generated signal at or near zero external bias. It can reduce dark-current shot noise and power dissipation, though readout design and junction impedance still matter.","ir_review"),
("reverse_bias_pd","Why reverse-bias an infrared photodiode?",
 "Reverse bias can reduce junction capacitance, speed carrier collection and support avalanche or field-engineered operation, but it can also increase dark current, tunneling and breakdown risk. Optimum bias is device-specific.","hot_review"),
("mesa_vs_planar","What is the difference between mesa and planar infrared photodiode processing?",
 "Mesa devices define junction geometry by etching isolated structures, exposing sidewalls that require good passivation. Planar approaches form junctions without deep mesa isolation or use shallow isolation, which can reduce exposed sidewall area but require different doping/process control.","hot_review"),
("sidewall_leakage","Why can mesa sidewalls increase dark current?",
 "Etched sidewalls expose surfaces where damage, dangling bonds and contamination can create surface states and leakage paths. Passivation, etch chemistry and junction placement determine how strongly sidewalls contribute.","hot_review"),
("buffer_layer","Why use buffer layers in HgCdTe heteroepitaxy?",
 "Buffer layers can accommodate lattice/thermal mismatch, control nucleation and reduce propagation of defects from a foreign substrate. They do not make mismatch disappear; buffer thickness, composition and growth quality affect residual dislocation density.","czt_hgcdte"),
("substrate_orientation","Why does substrate orientation matter in HgCdTe epitaxy?",
 "Crystallographic orientation changes surface reconstruction, step structure, growth kinetics and defect formation. The preferred orientation depends on epitaxy method, polarity control and device architecture; it is not merely a mechanical wafer choice.","czt_hgcdte"),
("hg_pressure","Why is mercury overpressure important during HgCdTe growth or annealing?",
 "Mercury has high vapor pressure and HgCdTe defect chemistry is sensitive to Hg chemical potential. Controlled Hg overpressure/reservoir conditions can suppress Hg loss and shift vacancy/defect equilibria during growth or annealing.","ir_review"),
("hg_vacancies","Why do mercury vacancies matter in HgCdTe?",
 "Mercury vacancies are important native defects in HgCdTe and can act as acceptor-like defects, affecting conductivity type, compensation and carrier concentration. Their concentration depends on composition and thermal/mercury chemical-potential history.","ir_review"),
("anneal_type_conversion","Can annealing change HgCdTe conductivity type?",
 "Yes. Annealing under controlled mercury chemical potential can change native-defect populations and compensation, which may change carrier concentration or conductivity type. The result depends on composition, dopants, time and temperature.","ir_review"),
("ohmic_contact","What makes a good ohmic contact to an infrared semiconductor detector?",
 "A good ohmic contact has low, approximately linear contact resistance over the operating bias range and does not dominate noise or inject unwanted carriers. Metal choice, surface preparation, doping and anneal all affect contact behavior.","ir_review"),
("schottky_contact","What is a Schottky contact and why can it complicate detector measurements?",
 "A Schottky contact is a rectifying metal-semiconductor junction with a barrier. If an intended ohmic contact is partly rectifying, two-terminal I-V curves, responsivity and noise can be dominated by contact barriers rather than bulk detector physics.","ir_review"),
("four_wire","Why use a four-terminal measurement on a low-resistance detector?",
 "A four-terminal measurement separates voltage sensing from current injection so lead/contact resistance contributes much less to the inferred sample resistance. It is valuable when detector resistance is comparable to contacts or wiring.","ir_review"),
("van_der_pauw","What is the van der Pauw method used for?",
 "The van der Pauw method extracts sheet resistance and Hall parameters from a thin, approximately uniform sample with suitable peripheral contacts and simply connected geometry. Geometry and contact-size assumptions must be respected.","ir_review"),
("hall_sign","What does the sign of Hall voltage tell you?",
 "Under a defined magnetic-field/current polarity convention, Hall-voltage sign indicates the dominant carrier type in a simple single-carrier material. Mixed conduction or parallel layers can make the interpretation nontrivial.","ir_review"),
("hall_factor","Why can Hall mobility differ from drift mobility?",
 "Hall mobility includes a Hall scattering factor that depends on carrier statistics and scattering mechanism. The simple mu_H=sigma*|R_H| need not equal the true drift mobility exactly, especially in nonideal or multicarrier material.","ir_review"),
("self_heating","Why can detector bias cause self-heating?",
 "Electrical power P=IV dissipated in the detector raises its temperature if thermal conductance to the heat sink is finite. Self-heating can change resistance, dark current, responsivity and noise, so bias sweeps should check thermal effects.","hot_review"),
("load_line","Why is the bias circuit or load line important for a photoconductor?",
 "The detector and bias/load network jointly set operating voltage/current. If detector resistance changes with illumination or temperature, the actual device bias can shift, so nominal supply voltage alone may not describe operating conditions.","ir_review"),
("current_vs_voltage_noise","How do current-noise ASD and voltage-noise ASD relate?",
 "A transfer impedance or device impedance converts current fluctuations to voltage fluctuations. The conversion is frequency dependent when capacitance or amplifier dynamics matter, so A/√Hz and V/√Hz cannot be compared without the relevant impedance/transfer function.","ir_review"),
("enbw","What is equivalent noise bandwidth?",
 "Equivalent noise bandwidth is the bandwidth of an ideal rectangular filter that would pass the same white-noise power as the actual filter. It depends on the transfer-function shape and is required when converting a flat ASD to integrated RMS noise.","ir_review"),
("onesided_twosided","What is the difference between one-sided and two-sided PSD?",
 "A two-sided PSD represents positive and negative frequencies, while a one-sided PSD folds negative-frequency power onto positive frequencies for real signals. Numerical PSD/ASD factors differ, so the convention must be stated.","ir_review"),
("rms_from_asd","How do you convert white ASD to RMS noise?",
 "For approximately white ASD e_n passed through a system with equivalent noise bandwidth B_ENBW, RMS noise is e_n*sqrt(B_ENBW). This relation fails if the ASD is strongly frequency dependent across the band.","ir_review"),
("lockin_phase","Why does lock-in reference phase matter?",
 "A lock-in resolves the signal into in-phase and quadrature components relative to its reference. Incorrect phase can move real signal into the quadrature channel or reduce an in-phase reading; magnitude R is phase-independent but has different noise statistics.","ir_review"),
("lockin_harmonic","Why does chopper waveform shape matter for lock-in calibration?",
 "A chopped optical signal is generally not sinusoidal. The lock-in detects a selected Fourier harmonic, whose amplitude depends on duty cycle and waveform. Converting demodulated voltage to optical modulation therefore requires the correct harmonic convention.","ir_review"),
("aliasing","Why does aliasing matter in detector-noise spectra?",
 "Frequency components above half the sampling rate can fold into the measured band unless sufficiently attenuated before digitization. Anti-alias filtering and sample rate are therefore part of a valid PSD measurement.","ir_review"),
("windowing","Why use a window function in an FFT noise measurement?",
 "Finite records cause spectral leakage when signals are not periodic within the sampled interval. Window functions reduce leakage at the cost of resolution and noise-bandwidth changes; PSD normalization must account for the chosen window.","ir_review"),
("averaging_psd","Why average multiple PSD estimates?",
 "A single periodogram has high statistical variance. Averaging independent or partially overlapped segments reduces estimator variance, but segment length, windowing and overlap determine frequency resolution and statistical independence.","ir_review"),
("photon_noise","What is photon shot noise in infrared detection?",
 "Photon arrival statistics produce fluctuations in detected photon number. For thermal radiation, bunching/background statistics can add terms beyond simple Poisson shot noise; the exact photon-noise model depends on occupation, bandwidth and optical modes.","hot_review"),
("read_noise","What is read noise in an infrared focal-plane array?",
 "Read noise is noise introduced by the ROIC/readout and digitization chain rather than photon generation in the absorber. It can include reset, amplifier, transistor and sampling noise and is often expressed as electrons RMS per read or equivalent input noise.","roic_review"),
("well_capacity","What is pixel well capacity?",
 "Well capacity is the maximum charge a pixel integration node can store before saturation/nonlinearity. It sets a dynamic-range limit together with read noise and integration time.","roic_review"),
("integration_time","How does integration time affect infrared FPA signal-to-noise?",
 "Longer integration collects more signal charge but can also accumulate dark current/background and may saturate the well. Noise terms scale differently with integration time, so optimum integration depends on scene, detector and readout.","roic_review"),
("operability","What does pixel operability mean in an infrared focal-plane array?",
 "Operability is the fraction of pixels meeting specified performance criteria such as responsivity, noise, dark current or uniformity. The numerical operability depends on the chosen thresholds and test conditions.","roic_review"),
("nonuniformity","What is nonuniformity correction in infrared imaging?",
 "Nonuniformity correction compensates pixel-to-pixel offset and gain differences using calibration data. It improves image uniformity but cannot restore saturated, unstable or fundamentally nonresponsive pixels.","roic_review"),
("mtf","What is MTF in an infrared imaging system?",
 "The modulation transfer function describes how spatial contrast is transferred versus spatial frequency. Detector pixel aperture, optics, sampling, motion and processing all contribute to system MTF.","roic_review"),
("radiance_irradiance","What is the difference between radiance and irradiance?",
 "Radiance is power per projected area per solid angle (and possibly per spectral interval), whereas irradiance is incident power per receiving area integrated over incoming directions. Converting between them requires optical geometry/solid angle.","mid_ir_platform"),
("photon_energy","What is the photon energy corresponding to wavelength lambda?",
 "Photon energy is E=hc/lambda. In practical infrared units, E[eV]≈1.23984/lambda[µm]. Wavelength itself is not an energy and must not be substituted directly into a bandgap equation.","ir_review"),
("photon_flux_power","How do you convert monochromatic optical power to photon flux?",
 "For monochromatic power P at wavelength lambda, photon flux is Phi=P/(hc/lambda)=P*lambda/(hc). Spectral or broadband sources require integration over wavelength with the appropriate spectral density.","ir_review"),
("emissivity","What is emissivity in infrared radiometry?",
 "Emissivity is the ratio of a surface's emitted spectral radiance to that of an ideal blackbody at the same temperature and wavelength/direction/polarization conditions. Real calibrators require wavelength- and geometry-dependent emissivity when precision matters.","mid_ir_platform"),
("view_factor","Why does source-detector geometry matter in blackbody calibration?",
 "The detector only receives radiation within the optical throughput defined by source area, distance/aperture, solid angle and optics. Blackbody temperature alone does not determine power at the detector.","mid_ir_platform"),
("fnumber","How does f-number influence infrared detector background?",
 "For an imaging system, lower f-number generally accepts a larger solid angle and increases optical throughput/background per detector area, subject to stops and vignetting. Photon-background noise and saturation can therefore depend strongly on f/#.","hot_review"),
("free_carrier_absorption","What is free-carrier absorption?",
 "Free-carrier absorption is intraband optical absorption by mobile electrons or holes, often increasing toward longer wavelengths. It can alter infrared transmission independently of interband band-edge absorption.","ir_review"),
("interference_fringe","Why do thin semiconductor samples show fringes in infrared spectra?",
 "Parallel interfaces can form a Fabry-Pérot etalon, producing interference fringes whose spacing depends on thickness and refractive index. Fringes are optical cavity effects, not automatically electronic absorption features.","mid_ir_platform"),
("substrate_transmission","Why must substrate transmission be considered in detector spectroscopy?",
 "A measured spectrum includes absorption/reflection from the active layer, substrate, coatings and interfaces. Substrate/reference measurements are often required before assigning spectral structure to the detector material.","ir_review"),
("thermal_detector_speed","Why are many thermal infrared detectors slower than photon detectors?",
 "Thermal detectors rely on heating/cooling of a finite thermal mass through a finite thermal conductance, giving a thermal time constant C/G. Photon detectors can respond on electronic carrier timescales, though readout and transport may still limit speed.","microbolometer"),
("bolometer_tradeoff","What is the bolometer sensitivity-speed tradeoff?",
 "Lower thermal conductance can increase temperature rise per absorbed power and thus sensitivity, but usually increases thermal time constant for a given heat capacity. Device design balances responsivity, noise and frame-rate requirements.","microbolometer"),
("pyro_chop","Why do pyroelectric detectors require changing radiation?",
 "Pyroelectric detectors respond to changes in temperature/polarization rather than a steady equilibrium temperature. They therefore need modulated or transient radiation and naturally reject a perfectly constant incident flux after thermal equilibrium.","ir_review"),
("thermopile_seebeck","How does a thermopile detect infrared radiation?",
 "Absorbed radiation heats one set of thermocouple junctions relative to reference junctions. The Seebeck effect produces a voltage proportional to the temperature difference; no semiconductor bandgap matching to the IR photon energy is required.","ir_review"),
("qcl_intersubband","Why are QCLs called intersubband lasers?",
 "Quantum cascade lasers use transitions between engineered quantized subbands within the conduction band (or valence-band analogs), not conventional electron-hole interband recombination. Their wavelength is set mainly by heterostructure design.","qcl_review"),
("icl_interband","Why are ICLs called interband cascade lasers?",
 "Interband cascade lasers use interband electron-hole recombination in a cascaded type-II heterostructure. They differ from QCLs, whose gain transition is intersubband.","mid_ir_sources"),
("dfg","What is difference-frequency generation in mid-infrared photonics?",
 "Difference-frequency generation is a second-order nonlinear optical process where two input frequencies generate radiation at their frequency difference. Phase matching and nonlinear susceptibility determine conversion efficiency.","mid_ir_platform"),
("opo","What is an optical parametric oscillator for infrared generation?",
 "An OPO uses nonlinear parametric conversion in a resonator to generate signal and idler waves from a pump. It can provide tunable infrared output when phase matching and cavity conditions are satisfied.","mid_ir_platform"),
]

EXTRA_TAXONOMY=[
("Compare photon detectors, thermal detectors and coherent/heterodyne receivers in infrared sensing.",
 "Photon detectors convert absorbed photons into electronic excitations; thermal detectors sense temperature changes caused by absorbed power; coherent/heterodyne receivers mix the optical field with a local oscillator and preserve phase/frequency information. Their noise, bandwidth and cooling requirements differ."),
("Group common infrared detector materials by approximate spectral niche.",
 "Typical SWIR choices include InGaAs and some HgCdTe compositions; MWIR includes InSb, HgCdTe, InAsSb and T2SLs; LWIR/VLWIR high-performance photon detection is commonly associated with HgCdTe, T2SLs and QWIPs, while microbolometers detect thermal radiation broadly without a band-edge-selected photon process."),
("List major noise categories in an infrared detector measurement.",
 "Important categories include Johnson thermal noise, shot noise, generation-recombination noise, 1/f or excess noise, photon/background noise, contact noise, readout/amplifier noise and digitization/sampling artifacts. The dominant term depends on device, bias, temperature and frequency."),
("List major causes of apparent responsivity error.",
 "Common causes include incorrect incident-power calibration, active-area error, wrong RMS/peak convention, chopper harmonic factors, spectral mismatch, optical transmission error, saturation, bias drift, contact effects and bandwidth/phase errors."),
("List measurements that distinguish bulk, surface and contact limitations.",
 "Useful discriminants include four-terminal resistance, multiple contact spacings, bias/temperature noise spectra, passivation/process splits, Hall measurements, spectral response, transients/frequency response, surface analysis and dummy/readout baselines."),
("List common dark-current mechanisms in narrow-gap photodiodes.",
 "Depending on device and bias, dark current may contain diffusion current, depletion-region generation-recombination, band-to-band tunneling, trap-assisted tunneling, surface leakage and contact-related injection. Temperature and bias dependence help separate them."),
("List common optical losses before photons create useful charge.",
 "Losses include Fresnel reflection, absorption in windows/coatings/substrates, vignetting, scattering, parasitic absorption outside the active region, incomplete absorber absorption and recombination before carrier collection."),
("List system quantities needed to compare two infrared imagers fairly.",
 "Compare spectral band, f-number/aperture, detector temperature, integration time/frame rate, NETD/noise, spatial resolution/MTF, dynamic range, calibration, scene/background and processing assumptions."),
("Distinguish HgCdTe, CdZnTe and CdTe in one answer.",
 "HgCdTe is the tunable narrow-gap infrared absorber alloy; CdZnTe is commonly a near-lattice-matched substrate for HgCdTe and also a high-energy-radiation detector material; CdTe is a binary II–VI material used in radiation detectors and as a constituent/buffer/substrate-related material."),
("Distinguish InGaAs, InP and InGaP.",
 "InGaAs is a ternary III–V absorber widely used in NIR/SWIR photodiodes; InP is a binary III–V substrate/platform often lattice matched to In0.53Ga0.47As; InGaP is a different ternary phosphide alloy and should not be substituted for InGaAs."),
("Distinguish NEP, D* and responsivity.",
 "Responsivity converts optical input to electrical signal; NEP is the optical power producing signal equal to noise under a stated spectral/bandwidth convention; D* normalizes spectral NEP by sqrt(active area) using cm² for Jones. They are related but not interchangeable."),
("Distinguish detector lifetime, electrical RC time constant and measured -3 dB response time.",
 "Carrier lifetime describes recombination kinetics, RC time constant describes an electrical pole, and measured -3 dB response can contain both plus transport, trapping, thermal and instrument poles. Equality is only justified when one identified pole dominates."),
]

EXTRA_DIAGNOSTICS=[
("A HgCdTe sample shows lower resistance after illumination. Does that by itself prove photoconductive gain?",
 "No. Illumination-induced resistance change demonstrates photoconductivity, but gain requires relating collected current to generated carriers and transit/lifetime physics. Contacts, heating and bias redistribution must be excluded."),
("A detector spectrum cuts off at 10 µm at 77 K. Can you state Cd fraction without a bandgap model?",
 "Not uniquely. Convert the cutoff to an approximate bandgap, then specify a temperature-dependent HgCdTe Eg(x,T) relation and cutoff criterion. Different empirical relations and measured criteria can shift inferred x."),
("An InGaAs detector responds beyond 1.7 µm. Is it necessarily miscalibrated?",
 "No. Extended-InGaAs compositions intentionally increase In content to reduce bandgap and extend wavelength response, usually with increased lattice mismatch and dark current. Calibration should still be checked before attributing the response."),
("A photodiode's D* improved when area was reduced. Does that prove the material improved?",
 "No. D* normalizes by sqrt(area), but perimeter/surface leakage, capacitance and readout coupling may scale differently from area. Device geometry can improve measured D* without a change in bulk material quality."),
("A noise ASD is flat from 2–20 kHz. Is it definitely Johnson noise?",
 "No. Readout noise, shot noise and other white processes can also be flat. Compare magnitude to sqrt(4kTR), use dummy/readout baselines, and test scaling with R, T, bias and current."),
("A detector's 1/f corner moved after passivation. What can you conclude?",
 "Only that the low-frequency noise spectrum changed. The cause could involve surface states, contacts, resistance/bias redistribution or other process effects. Use controlled comparisons before assigning a microscopic mechanism."),
("A Hall measurement gives p-type material but a device I-V looks n-type-like. Is one measurement wrong?",
 "Not necessarily. Junction structure, contacts, parallel conduction, surface inversion, multilayers or temperature differences can make device behavior differ from a simple bulk Hall interpretation."),
("A blackbody test gives twice the expected signal. What should be checked first?",
 "Check source temperature/emissivity, aperture and solid angle, active detector area, optical transmission, chopper harmonic/RMS conventions, amplifier gain, spectral integration and whether the detector/readout is linear."),
("An FTIR spectrum has periodic fringes. Should they be smoothed away before extracting cutoff?",
 "Not blindly. Fringes may be Fabry-Pérot interference carrying thickness/index information. Model or reference-correct them, and verify that smoothing does not move the physical band edge or hide spectral artifacts."),
("Cooling an LWIR detector reduces dark current but signal also drops. Is that impossible?",
 "No. Cooling can change carrier mobility/lifetime, photoconductive gain, junction collection, contact resistance, optical calibration and electronics. Dark-current improvement does not guarantee constant responsivity."),
("An nBn detector has low dark current. Does that prove the absorber has long lifetime?",
 "No. A barrier architecture can suppress majority-carrier and depletion-region currents even if absorber lifetime is modest. Lifetime needs independent or model-supported measurement."),
("A QWIP has low response at normal incidence. Is the material necessarily poor?",
 "No. Intersubband selection rules can make normal-incidence coupling weak even with good quantum wells. Grating or cavity coupling may be required before judging absorber quality."),
("A microbolometer shows high responsivity but slow response. Is that expected?",
 "It can be. High thermal isolation increases temperature rise per absorbed power but often increases thermal time constant. Heat capacity and thermal conductance jointly set the sensitivity-speed tradeoff."),
("A device has huge responsivity at one bias but D* falls. Why?",
 "If noise rises faster than responsivity, NEP worsens and D* falls. Bias can increase gain while also increasing 1/f noise, shot noise, contact injection or self-heating."),
("A spectral response peak shifts after adding an AR coating. Does that prove the semiconductor bandgap changed?",
 "No. A coating changes wavelength-dependent optical coupling and cavity interference. Separate optical-transfer changes from intrinsic band-edge changes using reference spectra or absorption measurements."),
("A detector has the correct Johnson-noise magnitude but a strong low-frequency excess. Is it Johnson-limited?",
 "Only in the frequency range where the measured spectrum matches the Johnson floor and other terms are negligible. At low frequency the excess term dominates, so 'Johnson-limited' must be frequency-qualified."),
("A calculated NEP uses integrated RMS voltage divided by spectral responsivity. What is missing?",
 "The RMS noise must be referred to a defined measurement bandwidth/transfer function. To obtain spectral NEP in W/√Hz, convert RMS noise to ASD using ENBW when the white-noise approximation is valid."),
("A device's spectral cutoff and Hall carrier concentration both change after anneal. Does one cause the other?",
 "Not necessarily. Cutoff mainly reflects band structure/composition while Hall concentration reflects free carriers and defect/dopant compensation. Annealing can affect both through different mechanisms."),
("A low-resistance detector is measured with two wires and long leads. What failure mode is likely?",
 "Lead and contact resistance can become a substantial fraction of measured resistance, bias and Johnson-noise estimates. Four-terminal measurement or contact de-embedding is preferable."),
("A lock-in reports X≈0 but R is large. Is there no signal?",
 "Not necessarily. The reference phase may be near quadrature, moving the signal into Y. Inspect X, Y and phase or use magnitude R, while accounting for magnitude noise bias at low SNR."),
("A detector is called 'uncooled' but uses a thermoelectric cooler. Is that terminology precise?",
 "Not really. 'Uncooled' often means no cryogenic cooler, but a TEC still controls temperature. Report actual detector temperature and stabilization method rather than relying on the label."),
("A vendor quotes D* without wavelength, temperature or area. Can it be compared rigorously?",
 "No. D* is condition dependent. At minimum require wavelength/band, temperature, area, bias, bandwidth/noise convention and whether responsivity/noise were measured under the same conditions."),
("A photoconductor shows faster response at higher bias. Does that prove lifetime decreased?",
 "No. Transit time, RC loading, contact behavior and field-dependent collection can change with bias. Optical lifetime should be separated from transport/readout poles."),
("A detector has a 100 kHz electrical bandwidth but is chopped at 10 Hz. Is that automatically optimal?",
 "No. Low-frequency 1/f or drift may dominate at 10 Hz even though the detector is fast. Modulation frequency should balance detector/electronics response, background rejection and noise spectrum."),
("A room-temperature InSb detector has very high dark current. Is that surprising?",
 "No. InSb has a very narrow bandgap, so thermally generated carriers are substantial near room temperature. Cooling is commonly used to reduce dark current/noise."),
("A CdZnTe crystal has a ~1.6 eV bandgap. Does that make it an LWIR photodetector?",
 "No. A ~1.6 eV gap corresponds to visible/near-IR photon energies and is far too large for intrinsic LWIR interband detection. CdZnTe's common detector role is X-ray/gamma detection or as an HgCdTe substrate."),
("A model says In0.53Ga0.47P when asked for InGaAs. What is wrong?",
 "InGaAs contains indium, gallium and arsenic, not phosphorus. The common InP-lattice-matched absorber is approximately In0.53Ga0.47As; InGaP is a different alloy."),
("A model says Jones has units S/cm/V. What is wrong?",
 "Jones is the conventional unit of specific detectivity: cm·√Hz/W. Siemens per centimetre per volt is dimensionally unrelated to D*."),
("A model adds wavelength to D*=sqrt(A)/NEP. When is that wrong?",
 "For spectral NEP already expressed in W/√Hz, the standard definition is D*=sqrt(A_cm²)/NEP_density. No wavelength factor belongs there. Wavelength enters only through other quantities such as responsivity if NEP has not already been computed."),
("A model interprets NASD as NASDAQ in a detector-noise discussion. What should it do?",
 "In detector characterization, NASD should be interpreted from context as noise amplitude spectral density, typically with units such as V/√Hz or A/√Hz. Unrelated acronym meanings should not override the technical context."),
]

CONTRASTIVE=[
("Is MCT in this model primarily medium-chain triglycerides?",
 "No. Within FIRM's infrared-detector scope, MCT defaults to mercury cadmium telluride (HgCdTe) unless the user explicitly establishes another context."),
("Is CZT copper zinc telluride in HgCdTe epitaxy?",
 "No. In this context CZT means cadmium zinc telluride, Cd1-yZnyTe."),
("Is InGaAs the same as InGaP?",
 "No. InGaAs is indium gallium arsenide; InGaP is indium gallium phosphide. They are different III–V alloys with different band structures and applications."),
("Is CdZnTe the conventional LWIR absorber in MCT focal planes?",
 "No. CdZnTe is commonly the substrate; HgCdTe is the tunable narrow-gap infrared absorber."),
("Does a larger D* mean larger noise?",
 "No. At fixed active area, larger D* corresponds to lower spectral NEP and therefore better normalized sensitivity."),
("Can D* be calculated from active area and spectral NEP without responsivity?",
 "Yes. If NEP is already a spectral quantity in W/√Hz, D*=sqrt(A_cm²)/NEP_density. Responsivity was needed upstream only if NEP had to be derived from electrical noise."),
("Does NEP have units of watts in every context?",
 "No. Spectral NEP has units W/√Hz, while an integrated RMS noise-equivalent power has units W and depends on bandwidth/transfer function."),
("Is a time constant always the minority-carrier lifetime?",
 "No. A measured time constant can arise from carrier lifetime, RC response, diffusion, trapping, thermal dynamics or instrument filtering."),
("Does passivation always reduce 1/f noise?",
 "No. Good passivation can reduce surface-related leakage/noise, but a specific process can also introduce damage, stress, interface states or altered contacts. Measure rather than assume."),
("Is MBE a chemical-vapor process?",
 "No. Molecular beam epitaxy uses directed atomic/molecular beams in high/ultrahigh vacuum. MOCVD is the metalorganic chemical-vapor process."),
("Is LPE the same as MBE?",
 "No. LPE grows crystal from a liquid solution/melt, whereas MBE deposits controlled beams in vacuum."),
("Is InSb a ternary alloy?",
 "No. InSb is binary indium antimonide."),
("Is HgCdTe a III–V semiconductor?",
 "No. HgCdTe is a II–VI alloy semiconductor."),
("Are bolometers photon detectors?",
 "Bolometers are thermal detectors: absorbed radiation changes temperature and a temperature-dependent property. They are not bandgap-selected photon detectors."),
("Is a flat noise spectrum proof of Johnson noise?",
 "No. Several white-noise sources can be flat. Magnitude and scaling with resistance, temperature, current and readout baseline are needed."),
("Is a 1/f spectrum proof of surface traps?",
 "No. Multiple bulk, surface, contact and readout mechanisms can produce low-frequency excess or approximate 1/f behavior."),
("Does an ideal dark I-V prove good spectral responsivity?",
 "No. Electrical leakage and optical absorption/collection are different performance dimensions."),
("Does cooling always improve responsivity?",
 "No. Cooling often improves dark current/noise but can change lifetime, mobility, gain, contacts and readout conditions; responsivity must be measured."),
("Does higher responsivity always mean lower NEP?",
 "Only if noise does not rise proportionally more. NEP equals noise ASD divided by matching responsivity."),
("Can you infer trap physics from one noise frequency?",
 "No. Mechanism identification requires spectral shape plus bias, temperature, process and readout controls."),
]

def _extra_fact_rows():
    old_facts=base.FACTS
    old_sources=base.SOURCES
    try:
        base.FACTS=old_facts+EXTRA_FACTS
        base.SOURCES={**old_sources,**EXTRA_SOURCES}
        return base.make_fact_rows()
    finally:
        base.FACTS=old_facts
        base.SOURCES=old_sources

def _extra_taxonomy_rows():
    rows=[]
    all_tax=base.TAXONOMY+EXTRA_TAXONOMY
    for i,(q,a) in enumerate(all_tax):
        for v in range(10):
            rows.append({"id":f"taxonomy_v21_{i}_{v}","category":"taxonomy",
              "messages":[{"role":"system","content":base.SYSTEM},{"role":"user","content":base.perturb(q,v%12)},{"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research","rights":"original"}})
    return rows

def _extra_diag_rows():
    rows=base.diagnostic_rows()
    for i,(q,a) in enumerate(EXTRA_DIAGNOSTICS):
        for v in range(8):
            rows.append({"id":f"diag_v21_{i}_{v}","category":"diagnostic_reasoning",
              "messages":[{"role":"system","content":base.SYSTEM},{"role":"user","content":base.perturb(q,v%12)},{"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research","rights":"original"}})
    return rows

def _contrastive_rows():
    rows=[]
    for i,(q,a) in enumerate(CONTRASTIVE):
        for v in range(6):
            rows.append({"id":f"contrast_{i}_{v}","category":"contrastive_correction",
              "messages":[{"role":"system","content":base.SYSTEM},{"role":"user","content":base.perturb(q,v%12)},{"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research","rights":"original"}})
    return rows

def build(out:Path):
    if out.exists(): raise FileExistsError(out)
    raw=_extra_fact_rows()+_extra_taxonomy_rows()+base.quantitative_rows()+_extra_diag_rows()+base.refusal_rows()+_contrastive_rows()
    by_prompt={}
    for row in raw:
        by_prompt.setdefault(row["messages"][1]["content"],row)
    allrows=list(by_prompt.values())
    rng=random.Random(991745);rng.shuffle(allrows)
    cats={}
    for x in allrows:cats.setdefault(x["category"],[]).append(x)
    train=[];valid=[]
    for cat,rows in cats.items():
        k=max(2,round(len(rows)*.08))
        valid+=rows[:k];train+=rows[k:]
    rng.shuffle(train);rng.shuffle(valid)
    out.mkdir(parents=True)
    for split,rows in (("train",train),("valid",valid)):
        (out/(split+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows))
    urls=sorted(set(base.SOURCES.values())|set(EXTRA_SOURCES.values()))
    manifest={
      "schema_version":"2.1",
      "rows":{"train":len(train),"valid":len(valid)},
      "categories":{c:sum(1 for r in allrows if r["category"]==c) for c in sorted(cats)},
      "foundational_fact_cards":len(base.FACTS)+len(EXTRA_FACTS),
      "diagnostic_unique_targets":20+len(EXTRA_DIAGNOSTICS),
      "taxonomy_unique_targets":len(base.TAXONOMY)+len(EXTRA_TAXONOMY),
      "contrastive_unique_targets":len(CONTRASTIVE),
      "source_urls":urls,
      "private_benchmark_imported":False,
      "exact_user_stress_prompts_imported":False,
      "status":"PROVISIONAL_RESEARCH_CLAIM_CHECKED_NOT_EXPERT_RELEASE_APPROVED",
      "train_sha256":hashlib.sha256((out/"train.jsonl").read_bytes()).hexdigest(),
      "valid_sha256":hashlib.sha256((out/"valid.jsonl").read_bytes()).hexdigest(),
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,indent=2))
    return manifest

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("data/processed/firm4b_v2_foundation_v2"))
    a=ap.parse_args();build(a.out)
