"""Short-reasoning / complete-answer supervision for FIRM-4B v2.3."""

THINKING_DEPTH_CASES=[
("A detector engineer asks for a comparison between HgCdTe and InSb. What should be reasoned about before answering?",
 "Both are direct-gap IR semiconductors, but the decision hinges on tunable versus fixed bandgap, spectral range, transport, thermal generation, cooling, and materials/process complexity. Cover those axes and end with use-case implications.",
 """HgCdTe (MCT) and InSb are both direct-gap infrared semiconductors, but they occupy different design spaces. MCT is a ternary II-VI alloy whose bandgap can be tuned by Cd fraction, so one material system can cover MWIR, LWIR, and beyond. InSb is a binary III-V with a fixed narrow gap and is primarily an MWIR detector material.

InSb offers exceptionally high electron mobility and mature cooled MWIR performance, but its small gap causes high thermal carrier generation, so low-noise operation normally requires cooling. MCT also supports high carrier mobility and strong absorption, but its cutoff and thermal behavior depend strongly on composition. That tunability is MCT's major advantage and also its fabrication burden: composition uniformity, Hg vapor pressure, surfaces, and substrate/growth control are demanding.

For a conventional cooled MWIR imager, either can be excellent. MCT becomes particularly attractive when the cutoff must be engineered or when LWIR/VLWIR response is required; InSb is attractive when a fixed MWIR band and very high transport performance are sufficient."""),
("A user asks what a hermetic TO header is made from. What distinction should be made before naming a metal?",
 "Separate the package outline from the metallurgy, then distinguish compression-seal steel headers from matched-seal controlled-expansion alloys such as Kovar or Alloy 52. Mention plating separately from structural material.",
 """TO is a package outline, not one universal material specification. Hermetic TO headers are commonly built from cold-rolled steel in compression-seal designs or from controlled-expansion alloys such as Kovar or Alloy 52 in matched glass-to-metal seals. Kovar is therefore a very normal choice in detector and optoelectronic packages.

The pins are typically sealed through glass or ceramic and may also use Kovar/Alloy 52 depending on the seal system. Nickel and sometimes gold are surface finishes for corrosion resistance, solderability, wire bonding, or welding; they are not necessarily the bulk header material. The exact alloy should be taken from the package drawing because two TO headers with the same outline can use different seal metallurgy."""),
("A user names KRS-5. What identity check should happen before discussing applications?",
 "KRS-5 is a thallium-halide optical crystal, not a Ge-As semiconductor. Establish the TlBr-TlI identity, then explain its broad IR transmission and its mechanical/toxicity limitations.",
 """KRS-5 is thallium bromoiodide, a mixed TlBr-TlI crystal used as a very broadband infrared optical material. It is not a Ge-As semiconductor alloy. Depending on grade and thickness, KRS-5 transmits from the visible/near-IR edge to roughly 40 µm, which made it useful for FTIR windows, ATR optics, liquid cells, and some IR polarizer substrates.

Its drawbacks are substantial: it is soft, can deform or cold-flow under mechanical load, and contains toxic thallium. It therefore needs careful mounting and handling. KRS-5 is chosen when unusually broad spectral transmission matters more than ruggedness or ease of handling."""),
("A user asks for the best adhesive for an HgCdTe die on a ceramic cold stage. What should the reasoning prioritize?",
 "There is no universal best epoxy. Prioritize thermal path, CTE stress, modulus, cure temperature, electrical isolation, contamination, bond-line control, and thermal-cycle qualification. Give classes, not unsupported brand certainty.",
 """Choose the die attach from package requirements rather than from one headline property. For HgCdTe on an alumina cold stage, the main variables are cure temperature, thermal conductivity, bond-line thickness, CTE mismatch, elastic modulus/compliance, shrinkage, outgassing, ionic contamination, moisture uptake, and whether the bond must conduct electricity.

A silver-filled epoxy can provide a strong thermal path and electrical conduction, but it may be unacceptable if the detector must be electrically isolated or if silver migration/contamination is a concern. An insulating, more compliant epoxy can reduce stress transfer during cooldown but usually has higher thermal resistance. Whichever class is selected, control the bond line and qualify it with repeated thermal cycling, microscopy, shear testing, and leakage measurements. The best material is the one that survives the actual package stack and temperature range, not simply the epoxy with the highest catalog thermal conductivity."""),
("A user asks for an HgCdTe wet etch. What should be decided before naming a chemistry?",
 "Name established bromine families, then distinguish mesa geometry, isotropy, surface stoichiometry, passivation compatibility, and electrical sidewall quality. Avoid treating visual smoothness as electrical proof.",
 """Bromine-based wet chemistries are established starting points for HgCdTe, including Br2/methanol and Br2/HBr families. The best choice depends on the objective. For mesa etching, compare etch rate, undercut, sidewall angle, mask compatibility, post-etch stoichiometry, roughness, and subsequent passivation—not just how fast the material dissolves.

Wet etching avoids plasma bombardment and can leave low-damage surfaces, but it is often relatively isotropic. Dry plasma etching gives better dimensional control for small pixels, but it can create sidewall damage and altered surface chemistry. The deciding measurements should be perimeter-scaled dark current, leakage, low-frequency noise, and passivation stability. A visually smooth sidewall is not sufficient evidence that the surface is electronically benign."""),
("A user asks about preventing mercury loss during Te-rich HgCdTe LPE. What is the core physical model?",
 "Treat it as Hg chemical potential and local vapor-pressure control, not just 'keep it sealed.' Explain reservoir/source, thermal gradients, free volume, dwell, and verification.",
 """Mercury loss in Te-rich HgCdTe LPE is fundamentally a chemical-potential and vapor-pressure problem. Even though Te-rich solutions operate at much lower equilibrium Hg pressure than Hg-rich melts, excessive Hg loss changes the solution composition and liquidus condition and can shift layer composition or defect chemistry.

Maintain the intended local Hg activity near the growth solution with covered or enclosed hardware and an appropriate Hg reservoir/source. Control reservoir, melt, and substrate temperatures, minimize unnecessary high-temperature dwell and gas-flow stripping, and avoid large cold regions that pull Hg away from the growth zone by condensation. A sealed ampoule is helpful but does not by itself guarantee the correct local Hg partial pressure. Verify control from composition, electrical properties, surface condition, and run-to-run reproducibility."""),
("A user asks whether MCT is direct or indirect gap. What fact must be protected?",
 "Detector-relevant HgCdTe is direct-gap. The important implication is strong interband absorption plus composition-tunable bandgap. Do not confuse narrow gap with indirect gap.",
 """HgCdTe is a direct-bandgap II-VI alloy for detector-relevant compositions. The conduction- and valence-band extrema occur at the same crystal momentum, so near-edge interband absorption does not require a phonon. That is one reason relatively thin HgCdTe absorbers can still couple strongly to infrared radiation.

Its second major optical advantage is tunability: changing Cd fraction changes the direct bandgap and therefore the cutoff wavelength over a very broad IR range. Calling MCT an indirect-gap material is incorrect and would lead to the wrong picture of its absorption physics."""),
("A follow-up says a comparison was incomplete. How should the model respond?",
 "Expand along new decision-relevant axes instead of apologizing or repeating the same definitions. Add structure, quantitative anchors where reliable, and an engineering conclusion.",
 """Expand the comparison materially. For two detector materials, cover at least band structure and cutoff, absorption, carrier transport, thermal generation/dark current, cooling requirements, growth/substrate constraints, surface/passivation issues, manufacturability, and the applications each one is best suited to. Quantitative anchors are useful when they are stable and well defined, but do not invent numbers just to make the answer look complete.

The answer should end with a decision-useful conclusion: for example, one material may be favored for a fixed cooled MWIR band while another is favored when cutoff engineering or LWIR operation is required. Completeness means adding the dimensions needed to make a technical choice, not simply adding more words."""),
]
