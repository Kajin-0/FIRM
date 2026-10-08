"""FIRM-4B v2.3 depth-and-precision curriculum.

All prompts are paraphrased training targets. Exact user regression prompts remain
outside the committed curriculum. Answers are intentionally more complete than
v2.2 so the model learns adaptive technical depth rather than one-sentence replies.
"""

SOURCES={
"schott_to":"https://www.schott.com/en-us/products/transistor-outline-packages-p1000294/technical-details",
"rmt_to66":"https://www.rmtltd.ru/products/components/packages-for-optoelectronics/to-66-header/",
"complete_hermetics":"https://www.completehermetics.com/hermetic-products-and-services/to-headers/",
"krs5_edmund":"https://www.edmundoptics.com/f/thallium-bromoiodide-krs-5-windows/39725/",
"hgcdte_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7282310/",
"hgcdte_processing":"https://pmc.ncbi.nlm.nih.gov/articles/PMC10490682/",
}

DEPTH_CASES=[
("to_header_materials",
 "For a hermetic optoelectronic TO package, what is the header usually made from and why?",
 """A hermetic TO header is normally a metal/glass structure, not a plastic carrier. The common header metals depend on the glass-to-metal seal design. Compression-seal headers often use cold-rolled low-carbon steel, while matched-seal headers commonly use controlled-expansion alloys such as Kovar (ASTM F15) or Alloy 52 so the metal expansion better matches the sealing glass. The pins may also be Kovar or Alloy 52. Nickel, nickel-gold, or related plating is then added for corrosion protection, solderability, wire bonding, and cap welding. In detector and laser packages, Kovar is therefore entirely plausible and often desirable when hermeticity and thermal-cycle stability matter. The exact alloy should be confirmed from the package drawing because TO outlines define geometry, not one universal metallurgy.""",
 "schott_to"),
("to66_construction",
 "Describe the usual construction of an optoelectronic TO-66 header rather than a plastic transistor package.",
 """In optoelectronic use, a TO-66 header is a metal hermetic platform. Commercial detector/laser versions are commonly offered with a cold-rolled-steel or Kovar base and metal pins, usually with nickel and often gold plating. The electrical feedthrough pins are insulated from the header by glass or ceramic seals, and the die or thermoelectric cooler mounts to the central metal platform. A metal cap is then resistance- or laser-welded to the header, often with an optical window when the package is used for a detector or emitter. So if a TO-66 detector assembly looks like a Kovar or plated-steel header, that is normal; the package outline itself does not imply molded plastic.""",
 "rmt_to66"),
("kovar_role",
 "Why is Kovar so common in hermetic detector packages and glass-to-metal feedthroughs?",
 """Kovar is an Fe-Ni-Co controlled-expansion alloy used because its coefficient of thermal expansion can be closely matched to borosilicate sealing glasses over the sealing and operating range. That reduces differential strain at the glass-to-metal interface during sealing, cooldown, and thermal cycling, which helps preserve hermeticity and electrical isolation. It also develops an oxide layer that can bond well to sealing glass. The tradeoff is that Kovar is relatively dense and has much poorer thermal conductivity than copper, so it is chosen primarily for seal compatibility and mechanical reliability rather than heat spreading. In an IR package, it is common to combine Kovar feedthrough/header regions with more thermally conductive local mounts or heat-spreader structures where needed.""",
 "schott_to"),
("matched_vs_compression",
 "Compare matched glass seals and compression glass seals in TO headers.",
 """A matched glass-to-metal seal minimizes thermal stress by choosing a metal and glass with similar thermal expansion; Kovar and Alloy 52 are common metal choices. A compression seal deliberately uses a metal with a higher expansion than the glass so that, after cooling from the sealing temperature, the metal places the glass into compression. Cold-rolled steel is a common TO-header material for that approach. Both can be hermetic, but they use different thermo-mechanical strategies. Matched seals are attractive when dimensional and thermal-cycle stability around the feedthrough is critical, while compression seals can be robust and economical when the geometry and glass system are designed for compressive loading. The package drawing and seal glass determine which material system is appropriate.""",
 "schott_to"),
("to_plating",
 "What does the nickel or gold plating on a TO header actually do?",
 """The structural header alloy and the surface finish have different jobs. The base alloy supplies mechanical strength and participates in the glass-to-metal seal; nickel plating provides a corrosion-resistant diffusion/barrier surface and improves process consistency. Gold may be added locally or over nickel when good wire-bondability, solderability, contact stability, or oxidation resistance is required. Some TO hardware instead uses nickel-silver or other finishes. Plating does not mean the whole header is made from that metal. For detector packaging, the finish also matters for die attach, cap welding, contamination control, and long-term contact resistance, so it should be treated as a process specification rather than a cosmetic coating.""",
 "schott_to"),
("krs5_identity",
 "What is KRS-5 in infrared optics, and what is it used for?",
 """KRS-5 is thallium bromoiodide, a mixed thallium-halide crystal used as an infrared optical material; it is not a Ge-As semiconductor alloy. Its unusual attraction is very broad transmission, roughly from the visible/near-IR edge near 0.7 µm out to about 40 µm depending on thickness and quality. It has been used for FTIR windows, liquid-cell windows, ATR components, and substrates for IR wire-grid polarizers. KRS-5 is relatively soft and can cold-flow, so mechanical mounting must avoid excessive stress. It is also toxic because it contains thallium, so handling controls are required. Its value is therefore optical bandwidth, not semiconductor detector action.""",
 "krs5_edmund"),
("krs5_compare",
 "Compare KRS-5 with ZnSe and Ge as infrared window materials.",
 """KRS-5 offers exceptionally broad spectral transmission, extending far beyond the normal LWIR band, which makes it useful in broadband spectroscopy. ZnSe is much more common for practical IR windows and lenses, with good transmission through the MWIR/LWIR and much better mechanical familiarity; it is widely used around 10.6 µm. Germanium is strong and widely used for thermal-imaging windows, but its high refractive index produces large Fresnel loss without antireflection coatings and its useful transmission is narrower than KRS-5. KRS-5 is comparatively soft, prone to cold flow, and toxic due to thallium. Thus KRS-5 is chosen when extreme bandwidth is worth the handling/mechanical penalties, while ZnSe or Ge is usually easier for rugged imaging hardware.""",
 "krs5_edmund"),
("mct_direct_gap",
 "Is HgCdTe a direct- or indirect-bandgap semiconductor, and why does that matter for IR detection?",
 """Hg1−xCdxTe is a direct-bandgap II-VI alloy for detector-relevant compositions. Its conduction- and valence-band extrema are at the same crystal momentum near the zone center, so interband absorption does not require a phonon to conserve momentum. That gives strong optical absorption near the band edge, which is advantageous for thin photon-absorbing detector structures. The key HgCdTe feature is that the direct bandgap is continuously tunable with Cd fraction and temperature: changing x moves the cutoff from short-wave IR through MWIR, LWIR, and even VLWIR regimes. That combination of direct absorption and composition-tunable gap is one reason HgCdTe remains such a versatile infrared detector material.""",
 "hgcdte_review"),
("mct_vs_insb_overview",
 "Give a technically complete comparison of HgCdTe and InSb as infrared detector materials.",
 """HgCdTe and InSb are both direct-gap infrared semiconductors, but they solve different design problems. HgCdTe is a ternary II-VI alloy, Hg1−xCdxTe, whose bandgap is continuously tunable with composition and temperature. That allows one material system to cover SWIR through MWIR, LWIR, and VLWIR. InSb is a binary III-V semiconductor with a fixed narrow gap, about 0.17 eV near room temperature, and is used primarily for MWIR detection.

HgCdTe's main advantage is spectral tunability plus strong direct absorption and favorable carrier transport. Its main cost is materials complexity: composition uniformity, Hg vapor pressure, surface/interface control, and growth on specialized substrates are difficult. InSb is compositionally simpler and has extremely high electron mobility, but its narrow fixed gap produces high thermal carrier generation, so high-performance devices are normally cooled.

For a conventional cooled MWIR imager, either can be excellent. HgCdTe becomes especially compelling when the required cutoff must be engineered, when LWIR/VLWIR coverage is needed, or when one technology family must span several spectral bands.""",
 "hgcdte_review"),
("mct_vs_insb_photonics",
 "How do the optical and photonic properties of HgCdTe and InSb compare?",
 """Both HgCdTe and InSb are direct-gap materials, so both can have strong interband absorption without phonon assistance. The major optical difference is bandgap control. InSb has a fixed material composition and therefore a relatively fixed intrinsic spectral cutoff for a given temperature. HgCdTe is an alloy, so changing Cd fraction changes the direct bandgap and therefore the absorption edge and detector cutoff over a very wide infrared range.

InSb is particularly well suited to MWIR photon detection and combines its narrow gap with very high electron mobility. HgCdTe can be engineered for MWIR, LWIR, or still longer wavelengths while retaining strong absorption. In both materials, refractive index, free-carrier absorption, temperature-dependent band structure, and carrier lifetime affect optical coupling and device response. The practical implication is that InSb is a strong fixed-band MWIR platform, whereas HgCdTe is a spectrally programmable material system whose composition becomes an optical design variable.""",
 "hgcdte_review"),
("mct_vs_insb_transport",
 "Compare carrier transport and thermal limitations in HgCdTe and InSb detectors.",
 """InSb is famous for very high electron mobility, which supports fast transport and can be attractive for high-speed MWIR devices. Its very narrow bandgap also means thermal carrier generation becomes severe as temperature rises, so cooled operation is standard when low dark current and high detectivity are required. HgCdTe also supports high carrier mobility, but the transport and intrinsic carrier concentration depend strongly on Cd fraction because the bandgap is composition dependent.

For HgCdTe, the thermal problem therefore depends on the selected cutoff: an MWIR composition can operate warmer than an LWIR/VLWIR composition at comparable performance because the larger bandgap suppresses intrinsic carriers and Auger generation more effectively. HgCdTe additionally offers band-structure and doping design flexibility unavailable in binary InSb. The trade is that achieving the intended transport properties requires tighter control of composition, defects, surfaces, and contacts.""",
 "hgcdte_review"),
("mct_vs_insb_growth",
 "Compare the materials-growth and fabrication challenges of HgCdTe and InSb.",
 """InSb is a binary III-V compound, so its bulk composition is fixed once stoichiometry is established. HgCdTe is a ternary II-VI alloy whose detector cutoff depends sensitively on Cd fraction, making spatial composition uniformity a first-order device parameter. HgCdTe processing is further complicated by high Hg vapor pressure, relatively weak Hg-Te bonding, surface stoichiometry sensitivity, and defect chemistry that depends on Hg chemical potential.

HgCdTe is commonly grown on CdZnTe or on alternative substrates using engineered buffer layers, with LPE, MBE, and related methods. InSb has its own lattice, defect, passivation, and contact challenges, but it does not require tuning a ternary alloy composition to set the cutoff. Consequently, HgCdTe offers far greater spectral design freedom but usually demands tighter growth/process control. In production, that difference affects uniformity, yield, calibration, and cost as much as the underlying detector physics.""",
 "hgcdte_review"),
("mct_cutoff",
 "Explain how composition and temperature set the cutoff wavelength of HgCdTe.",
 """The HgCdTe bandgap is a function of both Cd mole fraction x and temperature T, so a cutoff wavelength cannot be mapped to a unique composition unless the temperature and bandgap model are specified. A first optical conversion is Eg≈hc/λc, but that only gives the gap corresponding to the chosen cutoff convention. One then solves an empirical relation such as Eg(x,T) for x at the stated temperature.

Increasing Cd fraction generally increases the bandgap and shifts the cutoff to shorter wavelength; increasing Hg content does the opposite. Temperature also changes Eg, so the same physical layer has a different cutoff at 77 K and 300 K. For serious detector work, the answer should therefore report the model, temperature, cutoff definition, and physical root rather than presenting x as a temperature-independent constant.""",
 "hgcdte_review"),
("insb_identity",
 "What is InSb, and what detector properties follow from its band structure?",
 """Indium antimonide is a binary III-V semiconductor with a very narrow direct bandgap, about 0.17 eV near 300 K. The direct gap gives strong infrared interband absorption, while the very small electron effective mass contributes to exceptionally high electron mobility. Those properties make InSb an important high-performance MWIR detector material.

The same narrow gap is also its central limitation: thermal generation of carriers and associated dark current become large at elevated temperature. High-performance InSb focal-plane arrays are therefore normally cooled. Unlike HgCdTe, InSb's bandgap is not composition-tuned because it is a binary compound, so its spectral range is much less adjustable. Its strength is excellent MWIR transport and mature detector technology rather than broad spectral tunability.""",
 "hgcdte_review"),
("cdznte_role",
 "Why is CdZnTe commonly used as a substrate for HgCdTe detectors?",
 """CdZnTe is attractive because its lattice constant can be adjusted with Zn content to closely match HgCdTe over important detector compositions, reducing misfit strain and the dislocation density associated with heteroepitaxy. It is also chemically and structurally compatible with the II-VI material system. A low-defect, lattice-matched substrate is especially valuable for HgCdTe because dark current, minority-carrier lifetime, and uniformity are sensitive to crystalline defects.

The disadvantages are cost, wafer-size limitations, availability, and substrate defects of its own. Those economic and scaling limitations are why alternative substrates such as GaAs, Si, or sapphire are sometimes used with CdTe/CdZnTe buffer architectures. The substrate choice is therefore a trade between epitaxial perfection, wafer scale, thermal/mechanical behavior, and manufacturing cost.""",
 "hgcdte_review"),
("ingaas_depth",
 "Explain InGaAs composition, InP lattice matching, and why extended-InGaAs is different.",
 """InGaAs is the ternary III-V alloy InxGa1−xAs. Its bandgap and lattice constant both vary with composition, so 'InGaAs' does not imply one fixed material. The composition near In0.53Ga0.47As is closely lattice matched to InP and is the standard platform for low-defect telecom/SWIR photodiodes. That lattice match is one reason conventional InGaAs detectors are so manufacturable.

To push the cutoff farther into the infrared, manufacturers increase the In fraction. The smaller bandgap extends the wavelength response, but the alloy then moves away from lattice match to InP. Extended-InGaAs therefore requires strain/mismatch management and generally pays a dark-current and material-quality penalty. The important distinction is that lattice matching is composition specific, not a property of every InGaAs alloy.""",
 "hgcdte_review"),
("mct_vs_t2sl",
 "Compare HgCdTe with III-V type-II superlattices for LWIR photon detection.",
 """HgCdTe provides a direct, composition-tunable bulk bandgap with strong absorption and remains a benchmark material for high-performance MWIR/LWIR detection. Type-II superlattices use engineered III-V layer sequences, commonly InAs/GaSb-family structures, to create an effective narrow gap through band alignment and quantum confinement. Their attraction is compatibility with III-V epitaxy and the ability to engineer band structure through layer thickness and composition.

HgCdTe generally offers very strong absorption and mature high-performance detector physics, but its material growth, Hg handling, substrate cost, and surface control are difficult. T2SLs avoid mercury and can exploit III-V manufacturing infrastructure, yet they face their own challenges in minority-carrier lifetime, interface quality, diffusion, and absorption thickness. The practical comparison therefore depends on cutoff, operating temperature, dark-current mechanism, array uniformity, and fabrication ecosystem rather than a single headline metric.""",
 "hgcdte_review"),
("die_attach_complete",
 "Give a complete engineering framework for choosing an adhesive between an HgCdTe die and a ceramic cold stage.",
 """Start with the thermal and mechanical requirements rather than a brand name. The adhesive must survive the full temperature range and repeated thermal cycling while maintaining adhesion to both HgCdTe/backside metallization and the ceramic substrate. Important variables are cure temperature, thermal conductivity, bond-line thickness, elastic modulus, glass-transition behavior, CTE mismatch, shrinkage, outgassing, ionic contamination, moisture uptake, and whether the joint must be electrically conductive or insulating.

A silver-filled epoxy can give low thermal resistance and an electrical path, but it can create leakage or contamination problems if the package requires isolation. A more compliant insulating epoxy may reduce thermally induced stress at the cost of higher thermal resistance. Keep the bond line controlled and void-free, then validate on witness assemblies with thermal cycling, shear testing, microscopy, and electrical leakage measurements. The correct adhesive is the one that passes the package-level qualification, not simply the one with the highest bulk thermal conductivity.""",
 "hgcdte_processing"),
("wet_etch_complete",
 "Explain how to choose a wet etch for HgCdTe rather than naming one chemical.",
 """Bromine-based chemistries are established HgCdTe wet etchants, including Br2/methanol and Br2/HBr families, but the best choice depends on what the etch must accomplish. For mesa definition, evaluate etch rate, lateral undercut, sidewall angle, surface roughness, post-etch stoichiometry, mask compatibility, repeatability, and how the resulting surface responds to passivation. A chemistry that produces a visually smooth surface can still leave electronically active defects or Te-rich surface conditions that increase leakage.

Wet etching avoids ion bombardment and can be gentle, but it is often comparatively isotropic. Dry plasma etching gives better anisotropy and dimensional control for small pixels but can introduce sidewall damage and altered stoichiometry. The correct comparison is therefore electrical: perimeter-scaled dark current, surface leakage, noise, and passivation stability should decide the process, not nominal etch rate alone.""",
 "hgcdte_processing"),
("hg_loss_complete",
 "Give a complete process explanation for controlling mercury loss during Te-rich HgCdTe LPE.",
 """Mercury control is fundamentally a chemical-potential and vapor-pressure problem. Even in Te-rich LPE, where the equilibrium Hg partial pressure is much lower than for Hg-rich melts, excessive Hg loss changes the melt composition and liquidus condition and can shift the solid composition or defect chemistry. The process should therefore maintain the intended local Hg activity at the solution throughout equilibration and growth.

Practical controls include covered or enclosed growth hardware, an appropriate Hg reservoir or local Hg source, controlled free volume and gas flow, minimized unnecessary high-temperature dwell, and tight control of the reservoir, melt, and substrate temperatures. A sealed ampoule alone does not guarantee the correct local Hg partial pressure if large thermal gradients or cold condensation zones remove Hg from the growth region. Verify success from layer composition, electrical properties, surface condition, and run-to-run reproducibility.""",
 "hgcdte_processing"),
("passivation_noise_complete",
 "Why can passivation improve HgCdTe surface stability yet make low-frequency noise worse?",
 """Passivation can reduce surface recombination or chemically stabilize the surface while still introducing or activating fluctuating charge states. A dielectric such as ZnS can alter interface-state density, fixed charge, local band bending, stress, contamination, or contact-edge fields. Those changes may have little effect on DC resistance but strongly affect carrier-number or mobility fluctuations, producing larger 1/f noise.

To distinguish mechanisms, compare otherwise identical passivated and unpassivated controls and examine noise versus bias, frequency, temperature, device perimeter/area, and contact geometry. Contact-dominated noise often scales differently with bias and geometry than bulk generation-recombination noise, while interface-trap fluctuations may show strong surface/perimeter sensitivity and temperature dependence. The important point is that lower DC leakage or a visually better surface does not guarantee lower noise; passivation must be qualified with spectral-noise measurements.""",
 "hgcdte_processing"),
("noise_metrics_complete",
 "Explain NEP, detectivity, specific detectivity, and voltage-noise spectral density without conflating them.",
 """Voltage-noise spectral density, often written en in V/√Hz, describes the detector/readout noise referred to a voltage output per square-root bandwidth. It is not the same as an integrated RMS voltage; integrating a white spectral density over effective noise bandwidth gives approximately Vrms=en√B.

Noise-equivalent power (NEP) is the input optical power that produces signal-to-noise ratio of one in a 1-Hz output bandwidth, so NEP has units W/√Hz when expressed as a spectral quantity. Detectivity is the reciprocal, D=1/NEP. Specific detectivity D* normalizes detectivity to detector area and bandwidth, conventionally D*=√(AΔf)/NEP, with units cm·√Hz/W (Jones). Higher D* therefore means better sensitivity, not more noise. To compare devices honestly, responsivity, area, bandwidth convention, temperature, bias, and readout contribution must all be stated.""",
 "hgcdte_review"),
("gr_vs_1f_complete",
 "How do generation-recombination noise and 1/f noise differ in a photoconductor?",
 """Generation-recombination noise arises from stochastic creation and recombination of carriers. A single dominant lifetime produces a Lorentzian spectrum: approximately flat at frequencies well below the characteristic corner and rolling off above fc≈1/(2πτ). Multiple lifetimes can broaden the shape.

1/f noise instead rises roughly inversely with frequency over some low-frequency range and is commonly associated with traps, mobility fluctuations, surfaces, contacts, or other excess-noise mechanisms. The two can overlap, so a spectrum alone is not always enough. Measure noise versus bias, temperature, illumination, geometry, and frequency over a wide band. If a reproducible Lorentzian corner shifts consistently with carrier lifetime or temperature, that supports G-R noise; if the excess follows an approximate 1/f law and strong contact/surface scaling, a flicker-noise mechanism is more plausible.""",
 "hgcdte_review"),
("hall_device_complete",
 "Why can Hall measurements and a two-terminal detector measurement appear inconsistent?",
 """Hall measurements infer bulk or sheet carrier type, density, and mobility under a defined geometry and magnetic field. A two-terminal detector measurement includes contacts, lead resistance, junctions, surface conduction, nonuniform current paths, and any parallel layers. The two experiments therefore need not report the same effective resistance or even probe the same dominant carriers.

If a detector I-V looks rectifying while Hall data are cleanly p-type, first examine contact barriers, contact metallurgy, surface inversion/accumulation, heterostructure layers, and measurement geometry rather than assuming the Hall measurement is wrong. For low-resistance material, four-terminal resistance is useful to remove lead/contact series resistance. Temperature, magnetic-field reversal, current reversal, and thickness uncertainty should also be audited in the Hall extraction.""",
 "hgcdte_review"),
]

DEPTH_POLICY_CASES=[
("A specialist is asked to compare two detector materials. What makes the answer complete rather than merely correct?",
 """A complete comparison should identify the material class and band structure, then compare the axes that actually affect detector choice: spectral range/cutoff, absorption, carrier transport, dark-current mechanisms, operating temperature, growth/substrate constraints, passivation and processing, manufacturability, and typical device architectures. It should finish with the engineering implication: which material is favored under which conditions. Merely stating that one material is II-VI and the other III-V is factually correct but operationally incomplete."""),
("A user asks a simple material-definition question. Should a specialist answer with one phrase or an essay?",
 """Use proportional depth. A definition normally needs enough information to establish identity, material family, the key physical property that makes it relevant, and its main application or limitation. That is usually a few sentences, not one noun phrase and not a page-long lecture. Concision should remove filler, not remove the physical context the reader needs to use the fact correctly."""),
("A user asks an engineering recommendation. What information should the response contain?",
 """Lead with the most likely options or process families, then explain the governing tradeoffs, likely failure modes, and the variables that would change the recommendation. If the answer depends on temperature, geometry, electrical isolation, surface preparation, or downstream processing, state those dependencies explicitly. 'It depends' is only useful when followed by what it depends on and how to validate the choice experimentally."""),
("A follow-up indicates the previous technical comparison was too shallow. How should the next answer change?",
 """Expand across new technical dimensions instead of merely adding another sentence. For detector materials, that may mean bandgap, absorption, transport, temperature limits, noise, growth, processing, and device implications. Preserve the original correct statements, but turn them into a decision-useful comparison. The goal is additional structure and explanatory depth, not repetition or conversational filler."""),
]

def word_count(text:str)->int:
    return len(text.split())

PRECISION_CORRECTIONS=[
("TO-66 is fundamentally a molded plastic semiconductor package.",
 "Incorrect as a general statement. The classic TO-66 outline is a metal package, and optoelectronic TO-66 headers are commonly hermetic metal/glass assemblies using cold-rolled steel or Kovar-family alloys depending on the seal design. Molded plastic packages may share similar lead arrangements, but that does not define TO-66 metallurgy."),
("Kovar is unusual in TO detector headers; brass is the normal hermetic header metal.",
 "Incorrect. Kovar and related controlled-expansion Fe-Ni/Fe-Ni-Co alloys are common in hermetic matched glass-to-metal seals, while cold-rolled steel is common in compression seals. Brass is not the default material to assume for a hermetic TO detector header."),
("KRS-5 is a germanium-arsenic semiconductor alloy.",
 "Incorrect. KRS-5 is thallium bromoiodide, a mixed TlBr-TlI infrared optical crystal. Its role is as a broadband IR transmitting material, not as a Ge-As semiconductor detector alloy."),
("HgCdTe is an indirect-gap semiconductor, which is why thick absorbers are needed.",
 "Incorrect. Detector-relevant HgCdTe is direct-gap and has strong interband absorption. Its bandgap is tunable with Cd fraction and temperature, which is central to its use across MWIR, LWIR, and longer-wave detector bands."),
("InSb and HgCdTe differ mainly because one is III-V and the other II-VI.",
 "That classification is correct but incomplete. The engineering differences also include fixed versus composition-tunable bandgap, spectral coverage, thermal carrier generation, cooling requirements, carrier transport, growth and substrate constraints, surface/passivation behavior, and manufacturing complexity."),
("A visually smooth HgCdTe wet etch proves the sidewall is electrically low-damage.",
 "Incorrect. Optical morphology does not directly measure interface-state density, surface stoichiometry, trap populations, or leakage pathways. Electrical qualification should include perimeter-scaled dark current, bias dependence, noise, and passivation stability."),
("A sealed LPE ampoule guarantees the correct Hg activity at the HgCdTe growth surface.",
 "Incorrect. Sealing limits gross Hg escape but does not by itself fix local Hg chemical potential. Reservoir temperature and location, thermal gradients, free volume, condensation zones, gas flow, and equilibration determine the local Hg partial pressure near the growth solution."),
("The epoxy with the highest catalog thermal conductivity is automatically the best HgCdTe die attach.",
 "Incorrect. Die attach is a package-level thermo-mechanical and electrical optimization. Cure temperature, modulus, CTE mismatch, shrinkage, bond-line thickness, electrical isolation, outgassing, contamination, voiding, and thermal-cycle reliability can outweigh bulk thermal conductivity."),
]
