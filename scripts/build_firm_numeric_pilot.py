#!/usr/bin/env python3
"""Create a small public development benchmark with transparent numerical oracles.

These are synthetic idealized cases, not empirical measurements or a production
test set. Scientific expert review is still required. Never overwrite a release.
"""
import argparse
import math
import subprocess
from pathlib import Path

from firm_data import digest, sha256, write_json, write_jsonl

K_B = 1.380649e-23
Q = 1.602176634e-19
H = 6.62607015e-34
C = 299792458.0


def hsc_gap(x, temperature):
    return -.302 + 1.93*x - .810*x*x + .832*x**3 + 5.35e-4*temperature*(1-2*x)


def bisect_composition(gap, temperature):
    lo, hi = .15, .5
    if not hsc_gap(lo, temperature) <= gap <= hsc_gap(hi, temperature):
        raise ValueError("Gap outside pilot bracket")
    for _ in range(80):
        mid = (lo + hi) / 2
        if hsc_gap(mid, temperature) < gap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pilot_items():
    items = []
    def add(identifier, category, prompt, quantities, oracle, rubric):
        items.append({"schema_version": "1.0", "id": identifier, "category": category,
                      "family_id": identifier, "prompt": prompt,
                      "provenance": {"kind": "synthetic", "source": "scripts/build_firm_numeric_pilot.py",
                                     "review_status": "oracle_verified", "answers_public": True,
                                     "license": "repository licensing unresolved; original synthetic task"},
                      "grading": {"quantities": {name: {"value": value, "unit": unit,
                          "rtol": .005, "atol": 0.0} for name, (value, unit) in quantities.items()},
                          "oracle": oracle, "manual_rubric": rubric}})
    gap = H*C/(9.4e-6*Q)
    x = bisect_composition(gap, 85)
    warm_gap = hsc_gap(x, 150)
    add("numeric_pilot_001", "HgCdTe_composition_cutoff",
        "An ideal HgCdTe absorption edge at 85 K corresponds to a 9.4 um cutoff. Use ONLY the supplied empirical model "
        "Eg[eV] = -0.302 + 1.93*x - 0.810*x^2 + 0.832*x^3 + 5.35e-4*T[K]*(1-2*x), with 0.15 < x < 0.5. "
        "Infer x, then predict cutoff at 150 K for the same composition. Define your optical-edge assumption, check the root, "
        "and explain why the result need not equal a measured responsivity cutoff. Use h=6.62607015e-34 J*s, "
        "c=299792458 m/s, q=1.602176634e-19 C.",
        {"gap_85K": (gap, "eV"), "composition_x": (x, "1"), "cutoff_150K": (H*C/(warm_gap*Q)*1e6, "um")},
        "Eg=hc/(q*lambda); bracketed bisection in x; re-evaluate supplied Eg(x,T).",
        ["Define optical edge versus measured cutoff criterion; empirical model is not a derivation.",
         "Keep composition fixed with temperature; discuss inhomogeneity, absorption threshold and model uncertainty."])
    before, after = 25e-9/1e4, 6e-9/7e3
    add("numeric_pilot_002", "passivation_noise_detectivity",
        "At 100 Hz a photoconductor before passivation has voltage responsivity 1.0e4 V/W and measured voltage-noise "
        "amplitude density 25 nV/sqrt(Hz). After passivation these are 7.0e3 V/W and 6 nV/sqrt(Hz). Active area is "
        "0.40 mm^2 in both cases. Assume calibration, bias, temperature and readout transfer are comparable. Compute both "
        "NEP densities, both D* values in Jones and the after/before D* ratio. Assess improvement and propose two controls "
        "before attributing the change to surface traps.",
        {"nep_before": (before, "W/sqrt(Hz)"), "nep_after": (after, "W/sqrt(Hz)"),
         "dstar_before": (math.sqrt(.004)/before, "Jones"), "dstar_after": (math.sqrt(.004)/after, "Jones"),
         "dstar_ratio": (before/after, "1")},
        "NEP_ASD=e_n/Rv; 0.40 mm^2=0.004 cm^2; D*=sqrt(A_cm2)/NEP_ASD.",
        ["Distinguish ASD from integrated RMS and PSD; no extra bandwidth factor for density NEP.",
         "Do not prove trap reduction from one frequency; compare dark/readout controls, PSD shape, optics and filter bandwidth."])
    tau, rc, freq = 150e-6, 40e-6, 1200
    measured = round(1/math.sqrt((1+(2*math.pi*freq*tau)**2)*(1+(2*math.pi*freq*rc)**2)), 8)
    recovered = math.sqrt(1/(measured**2*(1+(2*math.pi*freq*rc)**2))-1)/(2*math.pi*freq)
    add("numeric_pilot_003", "frequency_response_deembedding",
        f"A calibrated source drives a linear photoconductor with one carrier-response pole in cascade with a known "
        f"readout RC pole of 40 us. Normalized measured amplitude at 1200 Hz is {measured}. The low-frequency amplitude "
        "is unity and there are no other poles under the stated model. Infer carrier tau after removing the readout pole, "
        "and compute the detector-only 3 dB frequency. Contrast this with a naive single-pole fit to the measured amplitude. "
        "List measurements that could falsify the two-pole model.",
        {"carrier_tau": (recovered, "s"), "detector_f3db": (1/(2*math.pi*recovered), "Hz")},
        "|H|^2=[(1+(2*pi*f*tau)^2)*(1+(2*pi*f*tau_RC)^2)]^-1; solve explicitly.",
        ["Amplitude poles multiply; de-embed magnitude and check phase.",
         "Bias/geometry, calibrated source and preamp sweeps can distinguish microscopic lifetime from extra poles."])
    n, mu, length, width, thick, bias, lifetime = 2e21, .9, 400e-6, 80e-6, 12e-6, .2, 3e-6
    rho = 1/(Q*n*mu)
    transit = length**2/(mu*bias)
    add("numeric_pilot_004", "Hall_transport_photoconductor",
        "A uniform single-electron-carrier layer has n=2.0e15 cm^-3 and mobility 9000 cm^2/(V*s). A rectangular "
        "photoconductor has contact spacing 400 um, width 80 um, thickness 12 um, and 0.20 V across the active layer. "
        "Assume ohmic contacts, Hall factor unity, negligible holes and low-field drift. A separate optical measurement "
        "gives effective lifetime 3.0 us. Calculate dark resistance, electron transit time and photoconductive gain. "
        "Explain what Hall data alone cannot establish and what breaks the low-field model.",
        {"dark_resistance": (rho*length/(width*thick), "Ohm"), "transit_time": (transit, "s"),
         "gain": (lifetime/transit, "1")},
        "n_cm^-3 * 1e6 -> m^-3; mu_cm2/Vs * 1e-4 -> m2/Vs; rho=1/(q*n*mu); R=rho*L/(Wt); tau_t=L^2/(mu*V).",
        ["Hall factor and single-carrier assumptions must be explicit; lifetime is independently supplied.",
         "Check field, contacts, heating, trapping, compensation and geometry uncertainty."])
    coefficient, white = 4e-16, 9e-18
    variance = coefficient*math.log(500/2) + white*(500-2)
    add("numeric_pilot_005", "noise_PSD_band_integration",
        "The one-sided voltage PSD in a characterized measurement chain is S_v(f)=A/f+S_white for 2 <= f <= 500 Hz, "
        "where A=4.0e-16 V^2 and S_white=9.0e-18 V^2/Hz. Assume ideal rectangular band integration with no other noise. "
        "Find the equality corner frequency and integrated RMS noise. Give the dimensions of A and explain why simply "
        "multiplying the ASD at 2 Hz by sqrt(498 Hz) is inappropriate. Explain how a real lock-in changes the integral.",
        {"corner_frequency": (coefficient/white, "Hz"), "rms_noise": (math.sqrt(variance), "V")},
        "fc=A/S_white; variance=A*ln(f_hi/f_lo)+S_white*(f_hi-f_lo); rms=sqrt(variance).",
        ["Integrate PSD, not ASD; A has V^2 dimensions for frequency in Hz.",
         "Real filter weighting needs |H(f)|^2 and its ENBW; no universal single bandwidth for colored noise."])
    lam, temperature, dlambda, area, omega, transmission, rv = 8e-6, 420, .08e-6, 2e-8, .015, .55, 8000
    radiance = 2*H*C*C/(lam**5*math.expm1(H*C/(lam*K_B*temperature)))
    power = radiance*dlambda*area*omega*transmission
    add("numeric_pilot_006", "blackbody_throughput_voltage",
        "Use a narrow-band approximation at 8.0 um for a 420 K ideal blackbody. Bandwidth is 0.080 um, active area "
        "2.0e-8 m^2 and accepted projected solid angle integral int(cos(theta)dOmega)=0.015 sr. Transmission is 0.55 "
        "and voltage responsivity is 8000 V/W, constant in the band. The cold reference has negligible radiance. "
        "Calculate spectral radiance per um, open-state minus closed-state optical power and voltage difference. "
        "Report state differences, not the fundamental Fourier component of a square-wave chop. Explain any cosine "
        "or pi factors and approximation limits. Use h=6.62607015e-34 J*s, c=299792458 m/s, kB=1.380649e-23 J/K.",
        {"radiance_per_um": (radiance*1e-6, "W/(m^2*sr*um)"), "modulated_power_difference": (power, "W"),
         "voltage_difference": (power*rv, "V")},
        "B_lambda=2hc^2/lambda^5/expm1(hc/lambda*kBT); P=B_lambda*dLambda*A*projectedOmega*transmission; deltaV=Rv*P.",
        ["Convert per-m to per-um radiance; projected solid angle already includes cosine, so no extra pi.",
         "Distinguish state difference from lock-in fundamental/RMS, finite-band integration, emissivity and warm background."])
    add("numeric_pilot_007", "absorption_thickness_uncertainty",
        "A homogeneous slab has absorption coefficient alpha=1000 cm^-1 at the wavelength of interest. Neglect reflection "
        "and interference. Find thickness for 95% absorbed photons. Compute absorption for thickness 20 um, and explain "
        "why a 20 um layer cannot be claimed to absorb >95% under these assumptions. Identify why absorbed fraction "
        "still need not equal external quantum efficiency.",
        {"thickness_95pct": (-math.log(.05)/1000*1e4, "um"), "absorption_20um": (1-math.exp(-2), "1")},
        "Absorbed fraction=1-exp(-alpha*t); alpha in cm^-1 requires t in cm.",
        ["Beer-Lambert assumptions and unit conversion must hold.",
         "Collection, surface recombination, reflection and near-cutoff alpha affect external QE."])
    add("numeric_pilot_008", "lockin_ENBW_noise_density",
        "A calibrated voltage measurement has a flat one-sided input noise ASD of 12 nV/sqrt(Hz). A single-pole "
        "low-pass filter has H(f)=1/(1+i*2*pi*f*tau) with tau=20 ms and unity DC gain. Assume the demodulation "
        "normalization is already accounted for. Compute ENBW=int_0^infinity |H(f)|^2 df and output RMS noise. "
        "Explain why ENBW differs from f_3dB and what changes with colored input noise or a higher-order filter.",
        {"enbw": (1/(4*.020), "Hz"), "rms_noise": (12e-9*math.sqrt(1/(4*.020)), "V")},
        "One-sided integral of [1+(2*pi*f*tau)^2]^-1 is 1/(4*tau); rms=ASD*sqrt(ENBW).",
        ["Define one-sided PSD and unity DC gain; no unaccounted mixer factors.",
         "Use actual filter order/transfer and integrate colored PSD explicitly."])
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("evals/firm_numeric_pilot_v1.jsonl"))
    args = ap.parse_args()
    manifest = args.out.with_name(args.out.stem + "_manifest.json")
    if args.out.exists() or manifest.exists():
        ap.error("Pilot release exists; create a new version")
    items = pilot_items()
    write_jsonl(args.out, items)
    write_json(manifest, {"schema_version": "1.0", "benchmark_version": "firm_numeric_pilot_v1",
                         "generation_date": "2026-10-05", "status": "public development pilot; expert review pending",
                         "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                         "source_script": "scripts/build_firm_numeric_pilot.py",
                         "source_script_sha256": sha256(Path(__file__)), "answers_public": True,
                         "assets": [{"path": str(args.out), "sha256": sha256(args.out),
                                     "records": [{"id": r["id"], "record_sha256": digest(r), "category": r["category"]} for r in items]}],
                         "leakage_check": {"status": "must be audited against every training release",
                                           "semantic_review": "pending; related core equations are expected training knowledge"}})
    print(f"Created {len(items)} synthetic pilot cases; no model results generated")


if __name__ == "__main__":
    main()
