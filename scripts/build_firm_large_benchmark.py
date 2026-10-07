#!/usr/bin/env python3
"""Build a large, independent FIRM quantitative benchmark without training leakage.

Two explicitly distinct tiers:
- known_family_parameter_holdout: same 22 families as formula-generated SFT, unseen numeric seeds.
- unseen_family_transfer: new physics equations NOT present in the original 22-family SFT generator.
The generated benchmark is immutable once released and is NEVER training input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
from pathlib import Path
from build_firm_quantitative_sft import FAMILIES as TRAIN_FAMILIES, case as train_family_case

K=1.380649e-23
Q=1.602176634e-19
H=6.62607015e-34
C=299792458.
PI=math.pi
VERSION="1.0"

def fmt(x):
    return f"{x:.9g}"

def close(x,y, rtol=1e-10):
    return math.isfinite(x) and math.isfinite(y) and abs(x-y)<=rtol*max(abs(x),abs(y),1e-40)

def unseen_case(family,r):
    """Return prompt, key, exact value, unit, explanation, independently checked invariant."""
    if family=="einstein_diffusion":
        mu=r.uniform(100,14000)
        T=r.uniform(50,340)
        v=mu*1e-4*K*T/Q
        return (f"Use the nondegenerate Einstein relation for a semiconductor with mobility {fmt(mu)} cm^2/(V s) at {fmt(T)} K. Compute diffusion coefficient in m^2/s.",
                "diffusion_coefficient",v,"m^2/s","D = μ k_B T/q, with mobility converted from cm²/(V s) to m²/(V s). This assumes nondegenerate carrier statistics.",
                lambda x: close(x*Q/(mu*1e-4*K*T),1))
    if family=="diffusion_length":
        D=10**r.uniform(-5,-2)
        tau=10**r.uniform(-9,-5)
        v=math.sqrt(D*tau)*1e6
        return (f"Assume one-dimensional minority-carrier diffusion with D={fmt(D)} m²/s and effective lifetime {fmt(tau)} s. Compute diffusion length in µm.",
                "diffusion_length",v,"um","L=sqrt(D τ), then convert metres to micrometres. Surface and injection effects are excluded.",
                lambda x: close((x*1e-6)**2,D*tau))
    if family=="quadrature_noise":
        e1=10**r.uniform(-10,-7)
        e2=10**r.uniform(-10,-7)
        v=math.hypot(e1,e2)
        return (f"Two statistically independent white voltage noise ASD sources are {fmt(e1)} and {fmt(e2)} V/sqrt(Hz). What is total voltage ASD in V/sqrt(Hz)?",
                "total_asd",v,"V/sqrt(Hz)","For uncorrelated sources, add PSDs: sqrt(e1²+e2²), not their ASD amplitudes linearly.",
                lambda x: close(x*x,e1*e1+e2*e2))
    if family=="nep_rms_to_dstar":
        area_mm2=10**r.uniform(-2,.8)
        bw=10**r.uniform(.1,3.2)
        rms=10**r.uniform(-12,-8)
        v=math.sqrt(area_mm2*.01*bw)/rms
        return (f"A detector with active area {fmt(area_mm2)} mm^2 has integrated RMS NEP={fmt(rms)} W in ENBW={fmt(bw)} Hz, assuming white noise. Calculate specific detectivity D* in Jones.",
                "detectivity",v,"Jones","D*=sqrt(A_cm² × ENBW)/NEP_RMS. Bandwidth appears because given NEP is an integrated RMS value, not a W/sqrt(Hz) density.",
                lambda x: close(x*rms/math.sqrt(area_mm2*.01*bw),1))
    if family=="shot_limited_nep":
        I=10**r.uniform(-10,-3)
        R=r.uniform(.03,2.8)
        v=math.sqrt(2*Q*I)/R
        return (f"Assuming Poissonian dark current shot noise dominates, a photodiode has dark current {fmt(I)} A and current responsivity {fmt(R)} A/W. Find the spectral NEP in W/sqrt(Hz).",
                "spectral_nep",v,"W/sqrt(Hz)","NEP=sqrt(2qI)/R_i. Assumes one-sided shot-noise PSD and no excess, Johnson or readout noise.",
                lambda x: close((x*R)**2,2*Q*I))
    if family=="photon_current":
        eta=r.uniform(.18,.92)
        flux=10**r.uniform(7,13)
        v=Q*eta*flux
        return (f"A unity-gain photodiode collects a fraction {fmt(eta)} of an incident flux {fmt(flux)} photons/s. Find mean photocurrent in A, assuming each detected photon yields one electron.",
                "photocurrent",v,"A","I=q eta Phi; this excludes dark current and any multiplication gain.",
                lambda x: close(x/(Q*eta*flux),1))
    if family=="quantum_limited_responsivity":
        lam=r.uniform(1,14)
        eta=r.uniform(.2,.95)
        G=r.uniform(1.1,12)
        v=eta*G*Q*lam*1e-6/(H*C)
        return (f"At wavelength {fmt(lam)} µm, a detector has quantum efficiency {fmt(eta)} and photoconductive gain G={fmt(G)}. Compute ideal current responsivity in A/W.",
                "current_responsivity",v,"A/W","R=η G q λ/(hc). Gain and quantum efficiency are separate factors; assumes linear response and well-defined collection.",
                lambda x: close(x*H*C/(eta*G*Q*lam*1e-6),1))
    if family=="reflected_absorptance":
        reflection=r.uniform(.02,.45)
        alpha=10**r.uniform(2,3.8)
        thickness=r.uniform(.2,18)
        v=(1-reflection)*(-math.expm1(-alpha*thickness*1e-4))
        return (f"A single-pass IR layer has front reflectance {fmt(reflection)}, absorption coefficient {fmt(alpha)} cm^-1, thickness {fmt(thickness)} µm, no back reflection. Find incident-power absorptance.",
                "absorptance",v,"1","Absorptance=(1−R)[1−exp(−αd_cm)] assuming reflection occurs before one absorbing pass and no interference.",
                lambda x: close(x/(1-reflection),-math.expm1(-alpha*thickness*1e-4)))
    if family=="planck_photon_occupation":
        lam=r.uniform(2,20)
        T=r.uniform(95,600)
        z=H*C/(lam*1e-6*K*T)
        v=1/math.expm1(z)
        return (f"For blackbody temperature {fmt(T)} K and photon wavelength {fmt(lam)} µm, calculate mean photon occupation number per mode (Bose–Einstein factor).",
                "photon_occupation",v,"1","n=1/[exp(hc/(λkT))−1]. This is per-mode occupation, not photon flux and has no wavelength/frequency Jacobian.",
                lambda x: close(v*math.expm1(z),1))
    if family=="planck_spectral_radiance":
        lam=r.uniform(3,15)
        T=r.uniform(180,650)
        meters=lam*1e-6
        v=(2*H*C*C/meters**5)/math.expm1(H*C/(meters*K*T))*1e-6
        return (f"An ideal blackbody is at {fmt(T)} K. Compute its wavelength-domain spectral radiance at {fmt(lam)} µm in W/(m² sr µm).",
                "spectral_radiance",v,"W/(m^2 sr um)","B_λ=[2hc²/λ⁵]/[exp(hc/(λkT))−1] in W/(m² sr m); multiply by 1e-6 to convert per metre to per µm.",
                lambda x: close(x/1e-6*meters**5*math.expm1(H*C/(meters*K*T))/(2*H*C*C),1))
    if family=="ft_ir_resolution":
        opd=r.uniform(.5,25)
        v=1/(2*opd)
        return (f"An ideal FTIR scans maximum optical-path-difference {fmt(opd)} cm on one side of zero. Under the convention δsigma=1/(2 OPDmax), find nominal wavenumber resolution in cm^-1.",
                "resolution",v,"cm^-1","Using the specified nominal-resolution convention δsigma=1/(2 OPDmax). Apodization changes effective resolution.",
                lambda x: close(x*2*opd,1))
    if family=="f3db_to_tau":
        frequency=10**r.uniform(2,6)
        v=1/(2*PI*frequency)*1e6
        return (f"A detector behaves as an isolated first-order low-pass with measured -3 dB frequency {fmt(frequency)} Hz. Find effective response time in µs.",
                "response_time",v,"us","τ=1/(2π f3dB), converted to microseconds. This need not equal minority carrier lifetime when electronics also contribute.",
                lambda x: close(x*1e-6*2*PI*frequency,1))
    if family=="noise_equivalent_electrons":
        asd=10**r.uniform(-13,-9)
        bw=10**r.uniform(-1,4)
        integration=r.uniform(1e2,1e5)
        rms=asd*math.sqrt(bw)*integration/Q
        return (f"A transimpedance sensor has input white current-noise ASD {fmt(asd)} A/sqrt(Hz), effective noise bandwidth {fmt(bw)} Hz, and integrates charge for {fmt(integration)} seconds in this ideal mathematical model. Compute equivalent RMS electron count when equivalent charge is I_rms × integration time.",
                "electrons_rms",rms,"electrons","Compute I_rms=i_n sqrt(ENBW); Q_rms=I_rms t, N=Q_rms/q. This is a stipulated equivalent-charge conversion, not a physical independent boxcar ENBW model.",
                lambda x: close(x*Q/(asd*math.sqrt(bw)*integration),1))
    if family=="drift_transit_time":
        length=r.uniform(3,100)
        mobility=r.uniform(150,20000)
        field=r.uniform(50,3000)
        v=length*1e-6/(mobility*1e-4*field)*1e9
        return (f"A uniform-field device has electrode spacing {fmt(length)} µm, drift mobility {fmt(mobility)} cm²/(V s), and electric field {fmt(field)} V/m. Find drift transit time in ns ignoring velocity saturation.",
                "transit_time",v,"ns","v_d=μE; t=L/(μE), with mobility in m²/(V s), L in m, and ns=1e9 s.",
                lambda x: close((x*1e-9)*mobility*1e-4*field/(length*1e-6),1))
    if family=="quantum_efficiency_from_responsivity":
        lam=r.uniform(1,12)
        eta=r.uniform(.18,.9)
        resp=eta*Q*lam*1e-6/(H*C)
        v=resp*H*C/(Q*lam*1e-6)
        return (f"A unity-gain photodiode at {fmt(lam)} µm has measured responsivity {fmt(resp)} A/W. Infer external quantum efficiency fraction, assuming one carrier per photon.",
                "quantum_efficiency",v,"1","η=R hc/(qλ). Assume no avalanche/photoconductive gain and matching spectral/optical definitions.",
                lambda x: close(x*Q*lam*1e-6/(H*C),resp))
    raise KeyError(family)

UNSEEN_FAMILIES=(
    "einstein_diffusion","diffusion_length","quadrature_noise","nep_rms_to_dstar",
    "shot_limited_nep","photon_current","quantum_limited_responsivity",
    "reflected_absorptance","planck_photon_occupation","planck_spectral_radiance",
    "ft_ir_resolution","f3db_to_tau","noise_equivalent_electrons",
    "drift_transit_time","quantum_efficiency_from_responsivity",
)

def format_row(id_,tier,family,prompt,key,value,unit,explanation):
    response_contract=(" Return ONLY strict JSON with keys quantities and answer. "
       "Within quantities use key '"+key+"' containing numeric value and string unit. "
       "No markdown or extra commentary.")
    return {"id":id_,"schema_version":VERSION,"category":"quantitative_physics",
            "family_id":family,"tier":tier,"prompt":prompt+response_contract,
            "provenance":{"kind":"synthetic","review_status":"oracle_verified",
                          "answers_public":False,"source":"original_programmatic_generator",
                          "license":"original"},
            "grading":{"quantities":{key:{"value":value,"unit":unit,"rtol":0.003,
                                          "atol":max(1e-30,abs(value)*1e-12)}},
                       "manual_rubric":[explanation,
                                        "No false measured-device claim; state physical assumptions.",
                                        "Dimensional units and magnitude must be consistent."],
                       "oracle":"deterministic independently invariant-checked original equation"}}

def build(out:Path, familiar_per_family:int=250, unseen_per_family:int=400, seed:str="",force:bool=False):
    if familiar_per_family<1 or unseen_per_family<1:raise ValueError("Positive counts required")
    if len(seed)<32:raise ValueError("Private test seed must have at least 128-bit entropy")
    if (out/"manifest.json").exists() and not force:
        raise FileExistsError("Locked benchmark exists; explicit --force-regenerate needed, and public checksum commitments must be updated")
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    out.chmod(0o700)
    manifest={"schema_version":VERSION,"seed_commitment_sha256":hashlib.sha256(seed.encode()).hexdigest(),
        "generator":"build_firm_large_benchmark.py",
        "benchmark_only":True,
        "prohibited_training_use":True,
        "review_status":"formula invariants checked, not yet expert reviewed",
        "interpretation":"Known families overlap synthetic SFT equation families; unseen families do not. No proof of generalization from parameter-only holdout.",
        "source":"new independently generated cases; no old E0 prompts or SFT text imported",
        "tiers":[]}
    seen=set()
    # Exclude any exact SFT/validation prompt content (minus response-format suffix).
    train_root=Path("data/processed/firm3_synthetic_quant_v1")
    for split in ("train","valid"):
        path=train_root/(split+".jsonl")
        if not path.exists():raise FileNotFoundError("Benchmark isolation requires SFT data: "+str(path))
        for line in path.read_text().splitlines():
            source=json.loads(line)["messages"][1]["content"]
            source=source.removesuffix(" Return only JSON with keys quantities and answer.")
            seen.add(hashlib.sha256(source.encode()).hexdigest())
    for tier, families,n in (
        ("known_family_parameter_holdout",TRAIN_FAMILIES,familiar_per_family),
        ("unseen_family_transfer",UNSEEN_FAMILIES,unseen_per_family)):
        file=out/(tier+".jsonl")
        with file.open("w",encoding="utf8") as f:
            for family in families:
                for i in range(n):
                    for retry in range(1000):
                        r=random.Random(f"holdout--{seed}--{tier}--{family}--{i}--{retry}")
                        if tier=="known_family_parameter_holdout":
                            question,value,unit,key,explanation,checker=train_family_case(family,r)
                        else:
                            question,key,value,unit,explanation,checker=unseen_case(family,r)
                        h=hashlib.sha256(question.encode()).hexdigest()
                        if h not in seen:break
                    else:raise ValueError("Exhausted unique prompt space: "+family)
                    if not checker(value):raise AssertionError((tier,family,i,value))
                    if not math.isfinite(value):raise AssertionError("nonfinite")
                    seen.add(h)
                    record=format_row(f"{tier}_{family}_{i:04d}",tier,family,question,key,value,unit,explanation)
                    f.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+"\n")
        file.chmod(0o600)
        b=file.read_bytes()
        manifest["tiers"].append({"name":tier,"families":list(families),
                                  "family_count":len(families),"rows":n*len(families),
                                  "path":file.name,"bytes":len(b),
                                  "sha256":hashlib.sha256(b).hexdigest()})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    (out/"manifest.json").chmod(0o600)
    print(json.dumps({"tiers":[{"name":x["name"],"rows":x["rows"],
                                     "family_count":x["family_count"],"sha256":x["sha256"]} for x in manifest["tiers"]]},indent=2))
    return manifest

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path.home()/".local/share/firm-private-bench/large_quant_v1")
    parser.add_argument("--seed-file",type=Path,default=Path.home()/".config/firm-private-bench/large_quant_seed")
    parser.add_argument("--familiar-per-family",type=int,default=250)
    parser.add_argument("--unseen-per-family",type=int,default=400)
    parser.add_argument("--force-regenerate",action="store_true")
    args=parser.parse_args()
    args.seed_file.parent.mkdir(parents=True,exist_ok=True)
    if not args.seed_file.exists():
        args.seed_file.write_text(secrets.token_hex(32)+"\n",encoding="ascii")
        args.seed_file.chmod(0o600)
    seed=args.seed_file.read_text().strip()
    build(args.out,args.familiar_per_family,args.unseen_per_family,seed=seed,force=args.force_regenerate)
