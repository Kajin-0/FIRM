"""Concise supervised reasoning traces for FIRM-4B v2.2.

These are original, deliberately short rationales. They are not private benchmark
prompts and are kept separate so reasoning supervision can be audited.
"""

THINKING_CASES=[
("Why is the statement 'HgCdTe is III-V because it can be grown by LPE' wrong?",
 "Material family is set by constituent periodic-table groups, not growth method. Hg and Cd are group II and Te is group VI. LPE is only a crystal-growth technique.",
 "HgCdTe is a II-VI alloy semiconductor. LPE does not make a material III-V; the same growth method can be used for different semiconductor families."),
("A measured HgCdTe cutoff is given but temperature is not. Should I calculate x immediately?",
 "Cutoff gives an approximate bandgap through hc/lambda, but HgCdTe Eg depends on both x and temperature. Without temperature and a specified Eg(x,T) relation, a precise composition is underdetermined.",
 "No. First state the temperature and HgCdTe bandgap model, and ideally the cutoff convention; then solve Eg(x,T)=hc/lambda for x."),
("A detector noise spectrum is flat above a corner. Is Johnson noise proven?",
 "Flat spectral shape only establishes a white-noise region. Johnson, shot, readout and other white terms can overlap. Magnitude and scaling with resistance, temperature, current and readout baseline discriminate them.",
 "No. Compare the measured white floor with sqrt(4kTR), a readout/dummy baseline, and its R/T/bias scaling before calling it Johnson-limited."),
("After passivation, 1/f noise rises but resistance is almost unchanged. What should I infer?",
 "A nearly unchanged DC resistance does not rule out altered fluctuation mechanisms. Passivation can change interface traps, dielectric charge fluctuations, surface damage, stress or contact-edge behavior without a large DC shift.",
 "Treat the change as evidence that the process affected low-frequency fluctuations, not proof of one mechanism. Use bias, temperature, geometry and matched process-control comparisons to separate surface/interface, contact and bulk causes."),
("A user asks for the best wet etchant for HgCdTe without specifying geometry. How should I answer?",
 "There is no universal best chemistry, but a useful answer should still name established families. Bromine/methanol and bromine/HBr are common wet approaches; geometry, surface stoichiometry, mask compatibility and downstream passivation determine the choice.",
 "Start with bromine-based wet-etch families such as Br2/methanol and Br2/HBr, then condition the recommendation on target depth/profile, mask, acceptable undercut, surface stoichiometry and passivation. Fine high-aspect-ratio mesas may favor dry etching despite plasma-damage risk."),
("Why can a sealed Te-rich HgCdTe LPE system still lose effective Hg control?",
 "Sealing limits gross escape but does not guarantee the correct local equilibrium Hg pressure at the growth solution. Reservoir temperature/location, thermal gradients, free volume, gas flow and equilibration determine local Hg activity.",
 "Because containment and chemical-potential control are not identical. Maintain the intended Hg partial pressure near the Te-rich solution with suitable reservoir/source and thermal design, and verify composition/reproducibility rather than assuming 'sealed' is sufficient."),
("A die-attach epoxy has high thermal conductivity. Is that enough to recommend it for an HgCdTe cold stage?",
 "Thermal conductivity lowers bond thermal resistance, but cooldown also creates mechanical stress. Cure temperature, modulus, CTE mismatch, bond-line geometry, electrical conductivity, outgassing and contamination can dominate reliability.",
 "No. Screen thermal conductivity together with cure temperature, CTE/modulus, bond-line thickness, electrical isolation needs, outgassing/contamination and repeated thermal-cycle performance."),
("A model is uncertain whether Willoughby Smith discovered the photoelectric or photovoltaic effect. What is the precise answer?",
 "The historical observation was that illumination changed selenium's electrical conductivity/resistance. That is photoconductivity. Calling it a photovoltaic cell or Einstein's photoelectric-effect discovery conflates distinct phenomena.",
 "Willoughby Smith's 1873 selenium work is associated with photoconductivity: selenium's conductivity increased under illumination."),
("A photoconductor's responsivity increases with bias while D* decreases. Is that inconsistent?",
 "D* depends on both signal response and noise. Bias can increase photoconductive gain while increasing 1/f, shot, contact or self-heating-related noise even more strongly.",
 "No. If noise ASD rises faster than responsivity, NEP worsens and D* falls even though responsivity increases."),
("A user misspells several words while asking about G-R noise in an MCT photoconductor. Should scope confidence fall sharply?",
 "The technical anchors MCT, photoconductor and G-R noise are sufficient to establish domain context despite spelling errors. Robustness should rely on concepts, not exact surface form.",
 "No. Interpret the request in-domain and explain generation-recombination noise, its characteristic Lorentzian form when a dominant lifetime exists, and how it differs from approximate 1/f excess noise."),
("A two-wire resistance measurement on a low-resistance detector disagrees with Hall-derived expectations. What check comes first?",
 "When sample resistance is comparable to leads or contacts, a two-wire result includes those parasitics. Hall and two-terminal device behavior also probe different structures.",
 "First de-embed lead/contact resistance or repeat with a four-terminal geometry. Then investigate contacts, parallel conduction and device structure before declaring the Hall result wrong."),
("A numerical HgCdTe question requires solving an empirical Eg(x,T) relation. What should the model do before arithmetic?",
 "The high-value reasoning step is selecting the correct equation, temperature, units and physical root. Numerical root-finding is deterministic and can be delegated to a calculator or Python tool.",
 "State the chosen Eg(x,T) relation and assumptions, convert wavelength to bandgap with consistent units, then use a numerical tool to solve for the physical x in [0,1] and sanity-check the result."),
]
