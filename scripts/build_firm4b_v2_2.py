#!/usr/bin/env python3
"""FIRM-4B v2.2 curriculum: process engineering, answer depth, calibration.

Builds on v2.1 without importing private benchmark prompts or user stress prompts.
Targets are conceptual/paraphrased so evaluation prompts remain held out.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import build_firm4b_v2_expanded as v21
from firm4b_v2_2_reasoning import THINKING_CASES

SYSTEM=(
"You are FIRM, the Focused Infrared Research Model: a compact specialist in infrared "
"photonics, semiconductor detector materials/devices, fabrication, packaging, "
"characterization, radiometry, noise and supporting physics. Answer in-scope questions "
"directly and with enough substance to be useful. Simple factual questions may be short; "
"engineering, diagnostic, fabrication and comparison questions should give concrete "
"options, governing tradeoffs, likely failure modes, and the missing information that "
"would materially change the recommendation. Do not replace an answer with generic caveats "
"when established engineering guidance is available. State assumptions, equations and units "
"when relevant. Correct false premises explicitly. Distinguish established facts from "
"inference and uncertainty. Do not invent measurements, citations, organizations, products, "
"materials or mechanisms. Decline wholly unrelated requests."
)

SOURCES={
"hgcdte_review":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7282310/",
"hgcdte_processing":"https://pmc.ncbi.nlm.nih.gov/articles/PMC10490682/",
"hgcdte_surface_etch":"https://hwjs.nvir.cn/en/article/id/a11a283f-e755-42a5-9f12-2baa36ab47f2",
"hgcdte_lpe_containment":"https://patents.google.com/patent/US4317689A/en",
"hgcdte_lpe_te_rich":"https://patents.google.com/patent/US4567849A/en",
"epotek_h20e":"https://www.epotek.com/docs/en/Datasheet/H20E.pdf",
"epotek_hirel":"https://www.epotek.com/wp-content/uploads/2021/01/HI-REL-Die-Attach-Adhesives.pdf",
"masterbond_ep29":"https://www.masterbond.com/tds/ep29lpsp",
"smith_1873":"https://www.nature.com/articles/007361c0",
}

# Conceptual targets deliberately differ from the user's exact stress prompts.
PROCESS_FACTS=[
("die_attach_selection",
 "What properties should be prioritized when selecting an adhesive between a narrow-gap IR detector die and a ceramic cold-stage substrate?",
 "For a fragile narrow-gap detector bonded to a ceramic cold stage, evaluate cure temperature, thermal conductivity, bond-line thickness, elastic modulus/compliance, CTE-mismatch stress, outgassing, moisture/ionic contamination, electrical conductivity or insulation, adhesion to both surfaces, and thermal-cycle reliability. A thin controlled bond line improves thermal conductance, but a very stiff joint can transfer excessive stress during cooldown. The best choice depends on whether the bond must also provide an electrical path and on the detector's allowable cure temperature.",
 "epotek_hirel"),
("die_attach_products",
 "Give examples of commercially available epoxy families that could be screened for cryogenic or IR-detector die attachment, and explain why screening is still required.",
 "Examples worth screening include thermally/electrically conductive silver-filled die-attach epoxies such as EPO-TEK H20E when an electrical path is acceptable, and low-outgassing electrically insulating cryogenic epoxies such as Master Bond EP29LPSP when isolation is required. These are examples, not automatic selections for HgCdTe: cure temperature, modulus, CTE stress, thermal conductance, contamination, bond-line thickness and repeated thermal cycling must be validated on the actual package.",
 "epotek_h20e"),
("wet_etch_family",
 "What wet-etch chemistry families are commonly encountered for HgCdTe processing?",
 "Common HgCdTe wet chemistries include bromine/methanol and bromine/HBr families; modified bromine/HBr formulations are also used to control surface condition. There is no universal best chemistry because etch rate, isotropy, surface stoichiometry/Te enrichment, roughness, mask compatibility and subsequent passivation all matter. For fine mesa definition, compare wet-etch surface quality against dry-etch anisotropy and damage rather than choosing only by nominal etch rate.",
 "hgcdte_surface_etch"),
("wet_vs_dry_etch",
 "How should an engineer choose between wet and dry etching for HgCdTe mesa processing?",
 "Wet etching can provide comparatively low ion-bombardment damage and simple processing but may be isotropic and composition/surface-chemistry sensitive. Dry plasma etching provides better anisotropy and dimensional control but can create sidewall damage, defects and altered surface stoichiometry. The choice should be driven by mesa geometry, pixel pitch, sidewall leakage, passivation compatibility and measured post-etch electrical performance.",
 "hgcdte_processing"),
("hg_activity_lpe",
 "What is the thermodynamic reason mercury control matters during HgCdTe liquid-phase epitaxy?",
 "Mercury has high vapor pressure and HgCdTe phase/defect equilibria are sensitive to Hg chemical potential. In Te-rich LPE the equilibrium Hg pressure is far lower than over Hg-rich solutions, which makes growth more manageable, but loss of Hg can still shift solution composition and liquidus conditions. Process control therefore focuses on maintaining the intended local Hg partial pressure/activity throughout equilibration and growth.",
 "hgcdte_lpe_te_rich"),
("hg_loss_controls",
 "What process levers reduce mercury depletion during high-temperature HgCdTe growth or annealing?",
 "For Te-rich HgCdTe LPE, maintain the intended Hg partial pressure near the growth solution using covered/contained growth hardware and an appropriate local Hg source or reservoir; minimize gas-flow stripping and unnecessary exposed free volume, establish equilibrium before growth, and control solution/substrate temperature tightly. Excess Hg loss changes solution composition and liquidus conditions, hurting composition control and reproducibility. Verify the result from layer composition and electrical/material properties rather than assuming a sealed or covered vessel alone solved it.",
 "hgcdte_lpe_containment"),
("hg_loss_false_buffer",
 "Does adding a CdTe buffer layer by itself prevent mercury volatilization from HgCdTe?",
 "No. A buffer layer can help heteroepitaxy, nucleation or lattice/defect management, but it does not by itself set mercury vapor pressure or Hg chemical potential. Mercury-loss control requires thermodynamic/vapor management such as a controlled Hg-containing environment and temperature/time control.",
 "hgcdte_review"),
("lpe_material_class",
 "Is liquid-phase epitaxy primarily a III-V growth method?",
 "No. LPE is a general epitaxial growth method from a liquid solution and has been used for multiple semiconductor families, including II-VI HgCdTe as well as III-V materials. Calling HgCdTe a III-V material because LPE is used is a taxonomy error.",
 "hgcdte_review"),
("smith_precision",
 "What did Willoughby Smith's 1873 selenium work establish?",
 "Willoughby Smith, an English electrical engineer, reported that selenium's electrical resistance changed strongly with illumination. The historically relevant phenomenon is selenium photoconductivity, not the later Einstein photoelectric-effect explanation and not the photovoltaic effect.",
 "smith_1873"),
("ingaas_lattice_caveat",
 "Is every InGaAs composition lattice matched to InP?",
 "No. Only a particular composition near In0.53Ga0.47As is closely lattice matched to InP. InGaAs is composition-tunable; extended-wavelength InGaAs deliberately departs from the lattice-matched composition and requires strain/mismatch management.",
 "hgcdte_review"),
("process_answer_depth",
 "How should a specialist answer a fabrication recommendation when there is no single universally best process?",
 "Give the leading process families or candidate choices first, then state the engineering tradeoffs and the variables that decide among them. 'It depends' is incomplete unless followed by what it depends on, how to measure the decision variables, and what failure modes to watch for.",
 "hgcdte_processing"),
]

PROCESS_DIAGNOSTICS=[
("An HgCdTe die survives room-temperature cure but cracks after repeated cooldowns. What should you investigate?",
 "Prioritize thermo-mechanical stress: CTE mismatch among die, adhesive and ceramic/metal stack, adhesive modulus and glass transition, bond-line thickness/uniformity, voids, die size and cooldown rate. Compare a more compliant adhesive or controlled thinner/thicker bond line as appropriate, inspect fracture origin, and thermal-cycle witness assemblies before changing the semiconductor process."),
("A thermally conductive silver epoxy gives excellent cooldown but the detector leakage rises after assembly. What are plausible causes?",
 "Check whether the conductive adhesive created an unintended electrical path, contaminated high-impedance surfaces, wicked toward contacts, or introduced cure/thermal stress. Also separate true detector leakage from package/readout leakage with isolated dummy structures and resistance-to-ground measurements."),
("Two HgCdTe wet etches give the same depth but one device lot has much higher sidewall leakage. What should be compared?",
 "Compare post-etch surface stoichiometry/roughness, residual contamination, undercut/profile, mask residue, passivation adhesion, plasma or chemical damage history, junction proximity to the sidewall, and perimeter scaling of dark current. Equal depth does not imply equal electrical surface quality."),
("A mesa etch looks smooth optically but dark current rises sharply. Does the smooth surface prove the etch is benign?",
 "No. Optical smoothness does not measure electronic surface damage, stoichiometric alteration or interface-state density. Check perimeter dependence, surface chemistry, passivation response, bias/temperature dependence and an unetched/process-control split."),
("An HgCdTe growth run loses apparent Hg despite using a sealed vessel. What could still be wrong?",
 "A sealed vessel prevents gross escape but does not guarantee the correct local Hg chemical potential. Check Hg reservoir amount/location and temperature, thermal gradients, hot-zone dwell, free volume, leaks, condensation at colder surfaces, melt composition and whether equilibrium was reached before growth."),
("After ZnS passivation, low-frequency noise rises while DC resistance barely changes. What hypotheses deserve priority?",
 "Possible causes include new interface/surface trap fluctuations, process-induced surface damage or contamination, stress-related defect activation, altered contact-edge behavior, or added dielectric charge fluctuations. Use bias scaling, temperature dependence, frequency shape, contact geometry and matched unpassivated/process-control devices to distinguish them."),
("A process engineer asks for the 'best' adhesive for a detector cold stage but provides no operating temperature. What should the answer do?",
 "Do not stop at 'it depends.' Give the main selection criteria and candidate adhesive classes, then state that operating temperature/thermal-cycle range, electrical isolation requirement, allowable cure temperature, die size and desired thermal resistance are the missing variables that determine the final choice."),
("A user asks for a chemical etchant but does not specify mesa geometry or passivation. What is a useful answer?",
 "State the established wet-etch families and their main tradeoffs, explain when dry etching becomes attractive, and ask or condition the recommendation on target depth/profile, pixel pitch, sidewall leakage tolerance, mask system and downstream passivation. Avoid presenting one chemistry as universally optimal."),
("A model spends hundreds of tokens reconsidering the same material classification and never reaches a final answer. What behavior is preferred?",
 "Reason once, resolve the key uncertainty, and answer. Repetition of the same internal check does not improve reliability. When generation budget is limited, preserve enough budget for a complete final response and state uncertainty explicitly instead of looping."),
("A model knows a process fact but is unsure of a product-specific recommendation. What should it do?",
 "Separate the levels of confidence: give the established physics and selection criteria confidently, offer product examples only when supported, label them as screening candidates rather than guaranteed choices, and identify the validation test needed for the actual package."),
]

CONTRASTIVE=[
("Because HgCdTe is grown by LPE, it is a III-V semiconductor.",
 "Incorrect. Growth method does not determine semiconductor family. HgCdTe is a II-VI alloy; LPE is simply one epitaxial growth technique."),
("A sealed ampoule guarantees no mercury-related defect changes during HgCdTe processing.",
 "Incorrect. A sealed enclosure limits escape but Hg chemical potential still depends on reservoir conditions, temperature gradients, free volume and equilibrium. Verify composition/defect-sensitive properties."),
("The smoothest-looking HgCdTe etch must produce the lowest sidewall leakage.",
 "Incorrect. Optical morphology is only one observable. Electronic surface damage, stoichiometry, contamination and passivation interface quality can dominate leakage."),
("If an adhesive is thermally conductive it is automatically suitable for an HgCdTe cold-stage bond.",
 "Incorrect. Thermal conductivity is only one criterion. Cure temperature, electrical conductivity, CTE mismatch, modulus, outgassing, contamination, bond-line control and thermal cycling can disqualify an otherwise conductive adhesive."),
("Willoughby Smith discovered the photovoltaic effect in selenium.",
 "Not precisely. Smith's 1873 result established light-dependent conductivity/resistance in selenium, i.e. photoconductivity. Do not substitute the photovoltaic effect or Einstein's later photoelectric-effect theory."),
("All InGaAs is lattice matched to InP.",
 "Incorrect. InGaAs near In0.53Ga0.47As is lattice matched to InP; other compositions, including extended-InGaAs, are not."),
]

def _rows_from_facts():
    rows=[]
    for i,(_,q,a,src) in enumerate(PROCESS_FACTS):
        for v in range(6):
            rows.append({"id":f"v22_fact_{i}_{v}","category":"process_engineering",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":v21.base.perturb(q,v%12)},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research_claim_checked",
                          "rights":"original","evidence_url":SOURCES[src]}})
    return rows

def _rows_from_diag():
    rows=[]
    for i,(q,a) in enumerate(PROCESS_DIAGNOSTICS):
        for v in range(5):
            rows.append({"id":f"v22_diag_{i}_{v}","category":"diagnostic_reasoning",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":v21.base.perturb(q,v%12)},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research","rights":"original"}})
    return rows

def _rows_from_contrastive():
    rows=[]
    for i,(q,a) in enumerate(CONTRASTIVE):
        for v in range(5):
            rows.append({"id":f"v22_contrast_{i}_{v}","category":"contrastive_correction",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":v21.base.perturb(q,v%12)},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research","rights":"original"}})
    return rows

def _rows_from_thinking():
    rows=[]
    for i,(q,reasoning,a) in enumerate(THINKING_CASES):
        for v in range(6):
            rows.append({"id":f"v22_think_{i}_{v}","category":"reasoning_trace",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":v21.base.perturb(q,v%12)},
                          {"role":"assistant","reasoning_content":reasoning,"content":a}],
              "metadata":{"review_status":"provisional_reasoning_trace","rights":"original",
                          "thinking_trace":True}})
    return rows

def _rewrite_system(rows):
    out=[]
    for r in rows:
        x=json.loads(json.dumps(r))
        x["messages"][0]["content"]=SYSTEM
        out.append(x)
    return out

def build(out:Path):
    if out.exists(): raise FileExistsError(out)
    inherited=(v21._extra_fact_rows()+v21._extra_taxonomy_rows()+v21.base.quantitative_rows()+
               v21._extra_diag_rows()+v21.base.refusal_rows()+v21._contrastive_rows())
    raw=_rewrite_system(inherited)+_rows_from_facts()+_rows_from_diag()+_rows_from_contrastive()+_rows_from_thinking()
    by_prompt={}
    for row in raw:
        by_prompt.setdefault(row["messages"][1]["content"],row)
    allrows=list(by_prompt.values())
    rng=random.Random(22022026);rng.shuffle(allrows)
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
    manifest={
      "schema_version":"2.2",
      "rows":{"train":len(train),"valid":len(valid)},
      "categories":{c:sum(1 for r in allrows if r["category"]==c) for c in sorted(cats)},
      "inherited_from":"firm4b_v2_1_expanded",
      "process_fact_targets":len(PROCESS_FACTS),
      "process_diagnostic_targets":len(PROCESS_DIAGNOSTICS),
      "new_contrastive_targets":len(CONTRASTIVE),
      "thinking_trace_targets":len(THINKING_CASES),
      "source_urls":sorted(set(v21.base.SOURCES.values())|set(v21.EXTRA_SOURCES.values())|set(SOURCES.values())),
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
    ap.add_argument("--out",type=Path,default=Path("data/processed/firm4b_v2_2_expanded_v1"))
    build(ap.parse_args().out)
