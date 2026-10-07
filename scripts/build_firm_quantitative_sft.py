#!/usr/bin/env python3
"""Generate an original, formula-verified quantitative infrared-detector SFT curriculum.

This is synthetic, programmatically checked instruction data—not human-reviewed
literature, and not a claim of scientifically calibrated real-detector behavior.
No evaluation prompts, gold answers or inherited legacy data are used.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path

K = 1.380649e-23
Q = 1.602176634e-19
HC_EV_UM = 1.2398419843320026
H = 6.62607015e-34
C = 299792458.0
WIEN = 2897.771955

SYSTEM = ("You are FIRM, an infrared photodetector engineering assistant. "
          "Use coherent equations, units, stated assumptions, and physical checks. "
          "Never infer a specific mechanism from insufficient measurements.")

def sf(x, sig=5):
    if not math.isfinite(x):
        raise ValueError("Nonfinite oracle")
    return f"{x:.{sig}g}"

def case(name, r):
    """Produce (user prompt, value, unit, key, explanation, independent checker)."""
    if name == "photon_energy":
        wavelength = r.uniform(1.2, 18)
        val = HC_EV_UM / wavelength
        return (f"A detector receives photons at {wavelength:.5g} µm. Compute single-photon energy in eV.",
                val, "eV", "photon_energy", f"E = hc/lambda = {HC_EV_UM:.10g} eV·µm / {wavelength:.5g} µm. This is energy per photon, not optical power.",
                lambda x: abs(x*wavelength-HC_EV_UM)/HC_EV_UM < 1e-12)
    if name == "cutoff_wavelength":
        eg = r.uniform(0.075, 0.75)
        val = HC_EV_UM / eg
        return (f"Assume an ideal absorption edge with energy gap {eg:.6g} eV. What cutoff wavelength in µm follows?",
                val, "µm", "cutoff_wavelength", f"lambda_c = hc/Eg = {HC_EV_UM:.10g}/{eg:.6g} µm. This is an ideal gap-equivalent edge; measured response cutoff depends on the chosen criterion.",
                lambda x: abs(x*eg-HC_EV_UM)/HC_EV_UM < 1e-12)
    if name == "photon_rate":
        power = 10**r.uniform(-10,-4)
        lam = r.uniform(1.5, 12)
        val = power * (lam*1e-6)/(H*C)
        return (f"A monochromatic beam provides {sf(power)} W at {sf(lam)} µm. Find incident photon rate in photons/s.",
                val, "photons/s", "photon_rate", "Ndot = P/E_ph = P lambda/(hc). Assumes monochromatic power and counts incident, not collected, photons.",
                lambda x: abs(x*H*C/(power*lam*1e-6)-1) < 1e-12)
    if name == "johnson_density":
        resistance = 10**r.uniform(1,5)
        T = r.uniform(65,330)
        val = math.sqrt(4*K*T*resistance)
        return (f"At {sf(T)} K, a {sf(resistance)} Ω resistor is measured open-circuit. Find its one-sided Johnson voltage ASD in V/sqrt(Hz).",
                val, "V/sqrt(Hz)", "voltage_noise_density", "e_n = sqrt(4 k_B T R). Thermal equilibrium, classical frequency limit and no excess noise are assumed; this is a density, not integrated RMS noise.",
                lambda x: abs(x*x/(4*K*T*resistance)-1)<1e-12)
    if name == "shot_density":
        current = 10**r.uniform(-10,-2)
        val = math.sqrt(2*Q*current)
        return (f"Assuming Poissonian current shot noise, find current ASD for {sf(current)} A average current.",
                val, "A/sqrt(Hz)", "current_noise_density", "i_n = sqrt(2 q |I|). The full Poisson assumption and single-sided spectral convention matter; technical excess noise is excluded.",
                lambda x: abs(x*x/(2*Q*current)-1)<1e-12)
    if name == "diode_responsivity":
        lam = r.uniform(1.1,9.5)
        eta = r.uniform(.25,.95)
        val = eta*lam/HC_EV_UM
        return (f"An ideal unity-gain photodiode has quantum efficiency {sf(100*eta)}% at {sf(lam)} µm. Calculate current responsivity (A/W).",
                val, "A/W", "current_responsivity", "R_i = eta q lambda/(hc) = eta lambda_um/(1.23984 eV·µm); assumes one collected electron per absorbed/registered photon and no internal gain.",
                lambda x: abs(x*HC_EV_UM/(eta*lam)-1)<1e-12)
    if name == "photocurrent":
        resp = r.uniform(.05,3)
        power = 10**r.uniform(-9,-3)
        val = resp*power
        return (f"A linear IR detector has current responsivity {sf(resp)} A/W and incident modulated power {sf(power)} W. Find signal current in A.",
                val, "A", "photocurrent", "I_ph = R_i P. Use the responsivity measured with the same optical definition (incident vs absorbed) and modulation convention.",
                lambda x: abs(x/(resp*power)-1)<1e-12)
    if name == "voltage_signal":
        resp = 10**r.uniform(2,6)
        power = 10**r.uniform(-12,-7)
        val = resp*power
        return (f"An IR sensor has measured voltage responsivity {sf(resp)} V/W. For {sf(power)} W optical modulation, calculate signal voltage in V.",
                val, "V", "signal_voltage", "V_sig = R_V P_mod. Assumes a linear detector, matching modulation/lock-in amplitude conventions and no saturation.",
                lambda x: abs(x/(resp*power)-1)<1e-12)
    if name == "nep_density":
        noise = 10**r.uniform(-10,-7)
        resp = 10**r.uniform(2,6)
        val = noise/resp
        return (f"At the measurement frequency, detector voltage ASD is {sf(noise)} V/sqrt(Hz), voltage responsivity {sf(resp)} V/W. Compute spectral NEP.",
                val, "W/sqrt(Hz)", "nep_density", "NEP_density = e_n/R_V. Both quantities must refer to the same frequency and electrical/readout conditions; do not multiply by bandwidth.",
                lambda x: abs(x*resp/noise-1)<1e-12)
    if name == "detectivity":
        area_mm2 = 10**r.uniform(-2,1)
        nep_density = 10**r.uniform(-13,-9)
        area_cm2 = area_mm2*.01
        val = math.sqrt(area_cm2)/nep_density
        return (f"Detector active area is {sf(area_mm2)} mm^2 and spectral NEP is {sf(nep_density)} W/sqrt(Hz). Calculate D* in Jones.",
                val, "Jones", "specific_detectivity", "D* = sqrt(A_cm2)/NEP_density. Convert mm² to cm² by multiplying by 0.01; Jones = cm sqrt(Hz)/W. No extra bandwidth factor for an NEP density.",
                lambda x: abs(x*nep_density/math.sqrt(area_mm2*.01)-1)<1e-12)
    if name == "noise_rms":
        noise = 10**r.uniform(-10,-7)
        bw = 10**r.uniform(.0,4)
        val = noise*math.sqrt(bw)
        return (f"White one-sided voltage ASD is {sf(noise)} V/sqrt(Hz) across effective noise bandwidth {sf(bw)} Hz. Calculate integrated RMS voltage noise.",
                val, "V rms", "rms_noise", "v_rms = e_n sqrt(ENBW). This only holds for flat PSD across the filter; use integral S_v(f)|H(f)|² df otherwise.",
                lambda x: abs((x/noise)**2/bw-1)<1e-12)
    if name == "snr":
        v = 10**r.uniform(-7,-3)
        noise=10**r.uniform(-10,-7)
        bw=10**r.uniform(0,3)
        val=v/(noise*math.sqrt(bw))
        return (f"Signal RMS amplitude is {sf(v)} V; white voltage ASD {sf(noise)} V/sqrt(Hz), ENBW {sf(bw)} Hz. Determine linear amplitude SNR.",
                val, "dimensionless", "snr", "SNR = V_signal,rms/(e_n sqrt(ENBW)). The signal and noise must share the same RMS reference and filtering.",
                lambda x: abs(x*noise*math.sqrt(bw)/v-1)<1e-12)
    if name == "rc_cutoff":
        R=10**r.uniform(2,6)
        capacitance=10**r.uniform(-11,-7)
        val=1/(2*math.pi*R*capacitance)
        return (f"An isolated first-order RC low-pass has R={sf(R)} Ω and C={sf(capacitance)} F. Find -3 dB cutoff in Hz.",
                val, "Hz", "cutoff_frequency", "f_3dB = 1/(2 pi RC); this is a readout pole and must not automatically be identified with carrier lifetime.",
                lambda x: abs(x*2*math.pi*R*capacitance-1)<1e-12)
    if name == "lifetime_cutoff":
        tau = 10**r.uniform(-7,-3)
        val=1/(2*math.pi*tau)
        return (f"Assuming a single-pole detector impulse response with time constant {sf(tau,8)} s, calculate its -3 dB frequency in Hz.",
                val, "Hz", "f3db", "For a true single-pole model f_3dB=1/(2 pi tau). Additional optical, electronics or trap poles invalidate direct carrier-lifetime interpretation.",
                lambda x: abs(x*2*math.pi*tau-1)<1e-12)
    if name == "beer_lambert":
        alpha=10**r.uniform(2,3.5)
        d_um=r.uniform(.8,15)
        val=-math.expm1(-alpha*d_um*1e-4)
        return (f"A uniform absorber has alpha={sf(alpha)} cm^-1 and thickness {sf(d_um)} µm. Ignoring reflections, what fraction is absorbed?",
                val, "dimensionless", "absorbed_fraction", "A = 1-exp(-alpha d), using d_cm=d_um×10^-4. This excludes Fresnel reflection, cavity effects and collection efficiency.",
                lambda x: (alpha*d_um*1e-4 > 20) if x >= 1 else abs(-math.log1p(-x)/(alpha*d_um*1e-4)-1)<1e-12)
    if name == "optical_density":
        od=r.uniform(.05,4)
        val=10**(-od)
        return (f"An optical attenuator has base-10 optical density OD={sf(od)}. Find power transmittance as a fraction.",
                val, "dimensionless", "transmittance", "T=10^-OD. This is an ideal definition at the specified wavelength and does not determine reflection vs absorption.",
                lambda x: abs(-math.log10(x)-od)<1e-12)
    if name == "wien":
        temp=r.uniform(90,900)
        val=WIEN/temp
        return (f"Use Wien displacement law for a blackbody at {sf(temp)} K. Find wavelength of peak spectral radiance per unit wavelength in µm.",
                val, "µm", "peak_wavelength", "lambda_peak T = 2897.771955 µm K for the wavelength-domain Planck spectrum. Frequency-domain peak is not the same wavelength.",
                lambda x: abs(x*temp/WIEN-1)<1e-12)
    if name == "hall_density":
        rh=10**r.uniform(-5,-2)
        val=1/(Q*rh)/1e6
        return (f"A single-carrier sample has Hall coefficient magnitude {sf(rh)} m^3/C. Under a one-carrier Hall factor 1 assumption, infer carrier density in cm^-3.",
                val, "cm^-3", "carrier_density", "n=1/(q |R_H|) in m^-3, divide by 10^6 for cm^-3. Compensation, multiple carriers and Hall factor change this inference.",
                lambda x: abs(x*1e6*Q*rh-1)<1e-12)
    if name == "mobility":
        sigma=10**r.uniform(-1,3)
        rh=10**r.uniform(-5,-2)
        val=sigma*rh*1e4
        return (f"Single-carrier material: conductivity {sf(sigma)} S/m and |R_H|={sf(rh)} m^3/C. Estimate mobility in cm²/(V s).",
                val, "cm^2/(V s)", "hall_mobility", "mu_H=sigma |R_H| in m²/(V s); multiply by 10^4 to report cm²/(V s). Assumes one dominant carrier and a consistent Hall factor.",
                lambda x: abs(x/(1e4*sigma*rh)-1)<1e-12)
    if name == "photoconductor_gain":
        mu=r.uniform(100,10000)
        voltage=r.uniform(.05,3)
        length_um=r.uniform(50,600)
        tau=10**r.uniform(-8,-5)
        transit=(length_um*1e-6)**2/(mu*1e-4*voltage)
        val=tau/transit
        return (f"Ideal photoconductor model: mobility {sf(mu)} cm²/(V s), electrode length {sf(length_um)} µm, voltage {sf(voltage)} V and carrier lifetime {sf(tau)} s. Compute photoconductive gain.",
                val, "dimensionless", "photoconductive_gain", "t_tr=L²/(mu V) using SI units, and G=tau/t_tr. This uniform-field drift estimate ignores contact, trapping and saturation physics.",
                lambda x: abs(x*transit/tau-1)<1e-12)
    if name == "psd_integral":
        white=10**r.uniform(-21,-15)
        low=r.uniform(10,300)
        high=r.uniform(2000,20000)
        val=math.sqrt(white*(high-low))
        return (f"Constant one-sided voltage PSD is {sf(white)} V²/Hz between {sf(low)} and {sf(high)} Hz with unity passband gain. Find band-integrated voltage RMS noise.",
                val, "V rms", "integrated_noise", "v_rms=sqrt(integral S_v(f)df)=sqrt(S0(f_high-f_low)). PSD must be integrated before the square root, unlike ASD.",
                lambda x: abs(x*x/(white*(high-low))-1)<1e-12)
    if name == "resistivity":
        resistance=10**r.uniform(1,4)
        l_mm=r.uniform(.3,5)
        w_mm=r.uniform(.2,3)
        thick_um=r.uniform(3,30)
        val=resistance*(w_mm*1e-3)*(thick_um*1e-6)/(l_mm*1e-3)
        return (f"Uniform bar: measured bulk R={sf(resistance)} Ω, length {sf(l_mm)} mm, width {sf(w_mm)} mm, thickness {sf(thick_um)} µm. Find resistivity in Ω·m.",
                val, "ohm*m", "resistivity", "rho=R(wt)/L with all dimensions converted to metres. Assumes a uniform current path and properly removed contact resistance.",
                lambda x: abs(x*(l_mm*1e-3)/(resistance*w_mm*1e-3*thick_um*1e-6)-1)<1e-12)
    raise KeyError(name)

FAMILIES = [
 "photon_energy","cutoff_wavelength","photon_rate","johnson_density",
 "shot_density","diode_responsivity","photocurrent","voltage_signal",
 "nep_density","detectivity","noise_rms","snr","rc_cutoff",
 "lifetime_cutoff","beer_lambert","optical_density","wien",
 "hall_density","mobility","photoconductor_gain","psd_integral","resistivity"
]

def generate(out: Path, train_per_family: int=120, valid_per_family: int=12, seed: int=20261006):
    out.mkdir(parents=True,exist_ok=True)
    manifest={"schema_version":"1.0","generator":"build_firm_quantitative_sft.py",
              "seed":seed,"source_type":"original_programmatic_formula_synthesis",
              "source_rights":"original generated prompts and solutions",
              "scientific_status":"symbolic/numeric equations programmatically cross-checked; NOT human reviewed",
              "benchmark_exclusion":"No E0 benchmark prompts, answers, or source records imported",
              "limitations":["Parameter-disjoint train/validation only, same formula families.",
                             "Numerical equation checks do not validate all real-device assumptions.",
                             "No improvement is claimed until pre/post heldout evaluation."],
              "family_count":len(FAMILIES),"families":FAMILIES,
              "train_per_family":train_per_family,"valid_per_family":valid_per_family,
              "outputs":[]}
    for split,count,tag in [("train",train_per_family,71),("valid",valid_per_family,97)]:
        path=out/(split+".jsonl")
        with path.open("w",encoding="utf-8") as f:
            for family in FAMILIES:
                for i in range(count):
                    r=random.Random(f"{seed}-{tag}-{family}-{i}")
                    user,value,unit,key,explain,check=case(family,r)
                    if not check(value):
                        raise ValueError(f"Failed independent oracle check {family}-{i}")
                    # Train strict structured results for machine-auditable science.
                    # Rounded value must still agree with analytic oracle to 3e-5 relative.
                    rounded=float(sf(value,4))
                    if abs(rounded-value)>6e-4*max(abs(value),1e-50):
                        raise ValueError(f"Formatting error {family}-{i}")
                    answer={"quantities":{key:{"value":rounded,"unit":unit}},
                            "answer":explain+" Approximate result: "+sf(value,4)+" "+unit+"."}
                    row={"id":f"firm_quant_v1_{split}_{family}_{i:04d}",
                         "messages":[{"role":"system","content":SYSTEM},
                                     {"role":"user","content":user+" Return only JSON with keys quantities and answer."},
                                     {"role":"assistant","content":json.dumps(answer,ensure_ascii=False,separators=(',',':'))}],
                         "metadata":{"family":family,"partition":split,
                                     "eligibility":"programmatically_verified_synthetic",
                                     "provenance":{"kind":"original_synthetic","license":"original",
                                                   "review_status":"programmatically_verified_not_human_reviewed"},
                                     "oracle":{"value":value,"unit":unit,"key":key},
                                     "template_id":family}}
                    f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
        manifest["outputs"].append({"path":path.name,"rows":count*len(FAMILIES),
                                    "bytes":path.stat().st_size,
                                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,default=Path("data/processed/firm3_synthetic_quant_v1"))
    a.add_argument("--train-per-family",type=int,default=120)
    a.add_argument("--valid-per-family",type=int,default=12)
    args=a.parse_args()
    if min(args.train_per_family,args.valid_per_family)<1:
        a.error("Counts must be positive")
    m=generate(args.out,args.train_per_family,args.valid_per_family)
    print(json.dumps({"families":len(FAMILIES),"outputs":m["outputs"]},indent=2))
