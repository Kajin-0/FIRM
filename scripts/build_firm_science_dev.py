"""Build 40 public scientific development scenarios; preserve every existing release.

Original scenarios use deterministic SI reference functions. No seed prompts or
answers are ingested. Their elementary equations are domain knowledge; numerical
training families are conservatively reserved from the reviewed seed release.
"""
import argparse
import math
import subprocess
from pathlib import Path
from firm_data import digest, sha256, write_json, write_jsonl
from firm_science_reference import C, H, KB, Q, HC_EV_UM, composition, enbw, gap, gap_dx, lorentzian_variance, planck, simpson


def development_items():
    items=[]
    def add(category, prompt, values, equation, rubric, family=None):
        identifier=f'science_dev_{len(items)+1:03}'
        quantities={k:{'value':v,'unit':u,'rtol':.003,'atol':0,'oracle':equation} for k,(v,u) in values.items()}
        items.append({'schema_version':'1.0','id':identifier,'category':category,'family_id':family or category,
            'prompt':prompt,'provenance':{'kind':'synthetic','source':'scripts/build_firm_science_dev.py',
            'review_status':'oracle_verified','answers_public':True,'license':'project rights unresolved; original synthetic development scenarios'},
            'grading':{'quantities':quantities,'oracle':equation,'manual_rubric':rubric}})
    model='Use the supplied model Eg[eV]=-0.302+1.93*x-0.810*x^2+0.832*x^3+0.000535*T*(1-2*x), 0.15<x<0.5. '
    constants='Use h=6.62607015e-34 J*s, c=299792458 m/s, q=1.602176634e-19 C, kB=1.380649e-23 J/K. '
    x=composition(10.6,95)
    add('HgCdTe_edge_consistency',model+'An absorption-edge fit gives 10.6 um at 95 K. A second edge estimate is 9.0 um at 165 K. Infer x from the first only, predict the second cutoff, and report measured-minus-predicted wavelength residual. Is this enough to reject compositional uniformity?',
        {'composition':(x,'1'),'predicted_warm_cutoff':(HC_EV_UM/gap(x,165),'um'),'warm_residual':(9-HC_EV_UM/gap(x,165),'um')},
        'Eg=hc/(q*lambda); independent Newton root; fixed-x warm gap; residual=measured-predicted',
        ['Temperature, cutoff criterion and empirical-model uncertainty precede a material-quality claim; one residual does not identify gradients.'])
    x=.24;t=100;dx=gap_dx(x,t);e=gap(x,t)
    add('HgCdTe_sensitivity',model+'At x=0.24 and T=100 K calculate dEg/dx, d(lambda_cutoff)/dT at fixed x, and the one-sigma composition uncertainty from an independent 0.04 um cutoff uncertainty alone using first-order propagation. State what uncertainties are omitted.',
        {'gap_slope_x':(dx,'eV'),'cutoff_temperature_slope':(-HC_EV_UM*.000535*(1-2*x)/e**2,'um/K'),'sigma_x':(.04*e**2/(HC_EV_UM*dx),'1')},
        'dEg/dx=1.93-1.62x+2.496x2-.00107T; lambda=hc/Eg; first-order derivative propagation',
        ['x is dimensionless; dEg/dx therefore has eV units; coefficient/model and temperature uncertainties excluded.'])
    add('HgCdTe_temperature_sign',model+'Two ideal uniform absorbers have x=0.20 and x=0.35 at 80 K. Calculate their Eg at 80 K and their Eg changes on warming to 140 K. Explain the sign using the supplied model, without extrapolating beyond its calibration.',
        {'gap_x020':(gap(.2,80),'eV'),'gap_x035':(gap(.35,80),'eV'),'delta_gap_x020':(.000535*60*.6,'eV'),'delta_gap_x035':(.000535*60*.3,'eV')},
        'Supplied model substitution; DeltaEg=.000535*DeltaT*(1-2x)', ['Positive slope in this bracket is empirical, not a universal semiconductor law.'])
    rho=120*100e-6*8e-6/600e-6
    add('four_terminal_transport','A dark bar has length 600 um, width 100 um and thickness 8 um. Four-terminal resistance is 120 Ohm; two-terminal resistance is 180 Ohm. With equal ohmic contacts and uniform current, infer resistivity, conductivity and resistance per contact. What control distinguishes contacts from current crowding?',
        {'resistivity':(rho,'Ohm*m'),'conductivity':(1/rho,'S/m'),'contact_resistance':(30,'Ohm')},'rho=R4*W*t/L; sigma=1/rho; Rc=(R2-R4)/2', ['Four-terminal voltage-probe geometry and current uniformity must be checked.'])
    add('geometry_uncertainty','Independent 1-sigma relative uncertainties for a resistivity inferred as rho=R*W*t/L are R:2%, W:3%, t:5%, L:1%. Nominal rho=0.004 Ohm*m. Find relative and absolute sigma_rho. Identify a correlated thickness-width calibration that would invalidate quadrature addition.',
        {'relative_sigma':(math.sqrt(.02**2+.03**2+.05**2+.01**2),'1'),'sigma_resistivity':(.004*math.sqrt(.0039),'Ohm*m')},'Independent log-derivative variances sum; .02²+.03²+.05²+.01²=.0039', ['First-order, independent Gaussian errors; correlated errors require covariance.'])
    rh=-.0002*12e-6/(80e-6*.4);n=1.15/(Q*abs(rh));mu=abs(rh)*800/1.15
    add('Hall_factor_transport',constants+'A 12 um layer carries +80 uA in +0.40 T; antisymmetrized electron-like Hall voltage is -0.200 mV. Conductivity is 800 S/m and Hall factor r_H=1.15. Use R_H=-r_H/(q*n), mu_H=abs(R_H)*sigma=r_H*mu_drift. Find R_H, n and drift mobility. Discuss sign conventions and contact-offset removal.',
        {'hall_coefficient':(rh,'m^3/C'),'carrier_density':(n,'m^-3'),'drift_mobility':(mu,'m^2/(V*s)')},'RH=VH*t/(I*B); n=rH/(q*abs(RH)); mu=abs(RH)*sigma/rH', ['Single carrier, low-field Hall model; polarity and offset correction explicit.'])
    n,p,mun,mup=2e21,8e21,.7,.2
    rh=(p*mup*mup-n*mun*mun)/(Q*(n*mun+p*mup)**2)
    add('two_carrier_Hall','A low-field semiconductor has electrons n=2e21 m^-3, mu_n=0.70 m^2/(V*s) and holes p=8e21 m^-3, mu_p=0.20 m^2/(V*s). Assume both Hall factors unity. Compute conductivity and R_H=(p*mu_p^2-n*mu_n^2)/(q*(n*mu_n+p*mu_p)^2). Explain why the Hall sign can differ from the numerically majority carrier sign.',
        {'conductivity':(Q*(n*mun+p*mup),'S/m'),'hall_coefficient':(rh,'m^3/C')},'Two-carrier low-field conductivity/Hall formula supplied; q=1.602176634e-19 C', ['Mobility squared weights Hall signal; single-carrier density inference is invalid.'])
    sigma=Q*(3e21*.5+7e20*.12)
    add('bipolar_bar_resistance','A uniform 300 um by 60 um by 10 um bar has n=3e21 m^-3, p=7e20 m^-3, electron mobility 0.50 and hole mobility 0.12 m^2/(V*s). Assume parallel low-field carrier conduction and ohmic contacts. Find bulk resistance and the error ratio R_electron_only/R_bipolar.',
        {'bulk_resistance':(300e-6/(sigma*60e-6*10e-6),'Ohm'),'electron_only_ratio':((3e21*.5+7e20*.12)/(3e21*.5),'1')},'sigma=q*(n*mu_n+p*mu_p); R=L/(sigma*W*t)', ['Neglecting holes increases inferred resistance here; density and mobility assumptions are supplied.'])
    tt=(500e-6)**2/(.8*.25)
    add('bias_gain_comparison','A 500 um photoconductor has drift mobility 0.80 m^2/(V*s), lifetime 4 us and active-layer bias 0.25 V. Doubling active bias leaves lifetime/mobility unchanged in an ideal low-field model. Find initial transit time, initial gain and gain ratio. Which bias-sweep evidence would falsify this extrapolation?',
        {'transit_time':(tt,'s'),'gain':(4e-6/tt,'1'),'gain_ratio':(2,'1')},'tau_t=L2/(mu*V); gain=tau/tau_t', ['Active voltage, uniform field and ohmic reinjection; heating and velocity saturation can invalidate gain scaling.'])
    ri=20e-6/2e-6;g=ri/(.6*Q*4e-6/(H*C))
    add('photoconductor_gain_from_photons','At 4 um, a low-flux photoconductor produces 20 uA signal for 2 uW incident power. Independently calibrated external generation probability per incident photon is 0.60, excluding electrical gain. Infer current responsivity and effective gain. Does gain identify a microscopic minority-carrier lifetime?',
        {'responsivity':(ri,'A/W'),'effective_gain':(g,'1')},'Ri=I/P; Ri=eta*g*q*lambda/(hc)', ['Generation efficiency, electrical gain and EQE are different; lifetime also needs transit/model information.'])
    tau=1/(2*math.pi*3200)
    add('phase_lifetime','An isolated first-order detector, with electronics/source response independently removed, has phase -45 degrees at 3200 Hz. Find tau and normalized magnitude at 6400 Hz. Explain why a trapping relaxation can produce the same pole as recombination.',
        {'time_constant':(tau,'s'),'magnitude_6400Hz':(1/math.sqrt(5),'1')},'phase=-atan(omega*tau); at -45 degrees omega*tau=1; |H(2fc)|=1/sqrt5', ['Hz-to-radians factor 2*pi mandatory; pole does not uniquely identify microscopic mechanism.'])
    rp=200*800/1000; sv=4*KB*rp*rp*(90/200+300/800)
    add('Johnson_network','Two independent resistors 200 Ohm at 90 K and 800 Ohm at 300 K are in parallel into an infinite-impedance voltage input. Compute equivalent resistance and one-sided voltage noise ASD. State why assigning the equivalent resistor 90 K is wrong.',
        {'equivalent_resistance':(rp,'Ohm'),'voltage_asd':(math.sqrt(sv),'V/sqrt(Hz)')},'Independent Norton PSDs 4kBT/R sum; multiply by R_parallel2', ['Uncorrelated classical Johnson sources; source temperatures remain distinct.'])
    si=2*Q*(40e-6+3e-6)+4*KB*80/20000+(2e-12)**2
    add('shot_Johnson_readout_budget','At 80 K a diode has 40 uA background photocurrent, 3 uA dark current, differential shunt resistance 20 kOhm and independent amplifier-current ASD 2 pA/sqrt(Hz). Signal responsivity is 1.2 A/W. Assume independent Poisson shot currents plus classical Johnson and white amplifier noise. Compute total current ASD, NEP density and RMS current in 250 Hz ENBW.',
        {'current_asd':(math.sqrt(si),'A/sqrt(Hz)'),'nep_density':(math.sqrt(si)/1.2,'W/sqrt(Hz)'),'rms_current':(math.sqrt(si*250),'A')},'SI=2q*(Ibackground+Idark)+4kBT/R+amplifier_ASD2; NEP=sqrt(SI)/Ri', ['Background noise contributes although useful modulation is separate; no double-counting amplifier or diode sources.'])
    add('PSD_ASD_sidedness','An instrument reports constant TWO-SIDED current PSD 3e-24 A^2/Hz for positive and negative frequencies. Compute the corresponding one-sided ASD and integrated RMS in the positive-frequency band 10 to 210 Hz. State what is held fixed when converting conventions.',
        {'one_sided_asd':(math.sqrt(6e-24),'A/sqrt(Hz)'),'rms_current':(math.sqrt(6e-24*200),'A')},'One-sided PSD=2*two-sided positive-frequency PSD; ASD=sqrt(PSD); variance=S_one*positive_bandwidth', ['Factor sqrt2 for ASD, not factor2; negative frequencies must not be integrated again.'])
    a=3e-16;floor=4e-18
    add('colored_noise_filtered_band','A one-sided voltage PSD is 3e-16/f + 4e-18 V^2/Hz when f is in Hz. An ideal analysis retains only 5..80 Hz. Find RMS voltage and the fraction of variance from the colored term. Explain why a single-frequency ASD is insufficient.',
        {'rms_voltage':(math.sqrt(a*math.log(16)+floor*75),'V'),'colored_variance_fraction':(a*math.log(16)/(a*math.log(16)+floor*75),'1')},'variance=A*ln(fhi/flo)+Swhite*(fhi-flo); no ASD integration', ['Finite lower limit essential for ideal 1/f; real filter changes weighting.'])
    fc=500;s0=8e-20;sw=1e-20;tau=1/(2*math.pi*fc)
    add('GR_spectral_inverse','A one-sided current PSD follows Swhite + S0/(1+(f/fc)^2). Independently measured white PSD is 1e-20 A^2/Hz. Total PSD is 5e-20 at 500 Hz and 2.6e-20 at 1000 Hz. Solve fc, S0 and correlation tau; integrate the GR component over all positive f. Can mean current replace S0?',
        {'corner':(fc,'Hz'),'plateau':(s0,'A^2/Hz'),'tau':(tau,'s'),'gr_variance':(lorentzian_variance(s0,tau),'A^2')},'After subtracting white: ratio=2.5=(1+4z)/(1+z), z=1; variance=S0*pi*fc/2', ['PSD amplitude is independently inferred, not inferred from mean current; phenomenological trap time.'])
    v1=lorentzian_variance(6e-18,.002,3,400);v2=lorentzian_variance(2e-18,.0002,3,400)
    add('multiple_GR_band','Independent one-sided voltage Lorentzians have S0=6e-18 V^2/Hz,tau=2 ms and S0=2e-18 V^2/Hz,tau=0.2 ms. Calculate total RMS in 3..400 Hz and the first component variance fraction. State what correlation would change.',
        {'band_rms':(math.sqrt(v1+v2),'V'),'first_variance_fraction':(v1/(v1+v2),'1')},'Each variance=S0/(2*pi*tau)*(atan(2*pi*tau*fhi)-atan(2*pi*tau*flo)); independent variances sum', ['Correlated processes require cross-spectra; RMS values do not add linearly.'])
    add('lockin_filter_order','Two identical unity-gain low-pass poles each have tau=15 ms and are cascaded. For flat one-sided demodulated input ASD 9 nV/sqrt(Hz), find ENBW and output RMS. Mixer normalization is already calibrated. Compare with one pole without introducing another mixer factor.',
        {'enbw':(enbw(.015,2),'Hz'),'rms_voltage':(9e-9*math.sqrt(enbw(.015,2)),'V')},'int0inf [1+(2*pi*f*tau)2]^-2 df=1/(8*tau)', ['Use transfer squared and stated pole count; one-sided post-demodulator input convention.'],family='ENBW_filter_transfer')
    add('unequal_pole_ENBW','A calibrated demodulated white voltage ASD 7 nV/sqrt(Hz) passes two unity-gain poles with tau1=5 ms and tau2=20 ms. Compute exact ENBW and RMS. Why is summing the individual ENBWs incorrect?',
        {'enbw':(1/(4*(.005+.020)),'Hz'),'rms_voltage':(7e-9*math.sqrt(10),'V')},'Partial fractions: int |H1*H2|2 df=1/[4*(tau1+tau2)]', ['Poles cascade; magnitude-squared transfer functions multiply.'],family='ENBW_filter_transfer')
    tau=.004
    variance=simpson(lambda f:(1e-16/f+2e-18)/(1+(2*math.pi*f*tau)**2),2,600)
    add('colored_noise_actual_filter','A one-sided voltage PSD 1e-16/f+2e-18 V^2/Hz exists only from 2 to 600 Hz. It passes H=1/(1+i*2*pi*f*0.004). Numerically compute output RMS. Compute the unfiltered RMS too. Explain why white-noise ENBW alone cannot normalize this colored spectrum.',
        {'filtered_rms':(math.sqrt(variance),'V'),'unfiltered_rms':(math.sqrt(1e-16*math.log(300)+2e-18*598),'V')},'Composite Simpson 8192 panels of PSD*|H|2; unfiltered analytic log integral', ['No spectrum is assumed outside the stated support; integration convention matters.'])
    delta=5e-9;ri=.9
    add('chopper_RMS_fundamental','Ideal 50% square-wave optical power switches between 2 and 7 nW. Current responsivity is 0.90 A/W and frequency response is flat. Find optical mean power, signal-current fundamental PEAK and RMS amplitudes, and the responsivity falsely inferred using fundamental RMS current divided by 5 nW state difference.',
        {'mean_power':(4.5e-9,'W'),'fundamental_peak_current':(ri*2*delta/math.pi,'A'),'fundamental_rms_current':(ri*math.sqrt(2)*delta/math.pi,'A'),'naive_responsivity':(ri*math.sqrt(2)/math.pi,'A/W')},'Square-wave difference dP: fundamental peak=2*dP/pi; RMS=peak/sqrt2', ['State difference, DC mean and sinusoidal RMS are distinct; lock-in calibration must match the chosen convention.'])
    add('Norton_voltage_responsivity','A detector is explicitly modeled as small-signal photocurrent source Ri=0.65 A/W in parallel with 300 Ohm. Its AC bias supply is grounded and an 1100 Ohm load shunts the output. Find load voltage responsivity and voltage signal for 40 nW modulation. Neglect reactive elements.',
        {'voltage_responsivity':(.65*300*1100/1400,'V/W'),'signal_voltage':(.65*300*1100/1400*40e-9,'V')},'Norton source sees Rd||RL; Rv=Ri*(Rd||RL)', ['A/W multiplied by Ohm gives V/W; a dimensionless voltage divider alone does not.'])
    d1=math.sqrt(.09)/(4e-12);d2=math.sqrt(.36)/(10e-12)
    add('area_normalized_sensitivity','Device A has active area 9 mm^2, NEP density 4 pW/sqrt(Hz); device B has area 36 mm^2, NEP density 10 pW/sqrt(Hz). Compute both D* in Jones and B/A ratio. Which has better area-normalized sensitivity despite collecting more total area? Do not add a bandwidth factor to a density.',
        {'dstar_A':(d1,'Jones'),'dstar_B':(d2,'Jones'),'ratio_B_A':(d2/d1,'1')},'1 mm2=.01 cm2; Dstar=sqrt(A_cm2)/NEP_ASD', ['Larger collecting area and normalized Dstar are separate metrics; matching frequency/bias/temperature required.'])
    f=1800;w=2*math.pi*f;rc=.00006;dt=.00012;amp=1/math.sqrt((1+(w*rc)**2)*(1+(w*dt)**2))
    add('cable_RC_deembedding',f'A detector with a calibrated carrier pole is measured through total R=20 kOhm and C=3 nF. At 1800 Hz the normalized amplitude is {amp:.10f}. Assume source flat and exactly two cascaded poles. Infer RC time and carrier time. Predict total phase at that frequency and give a cable-capacitance control.',
        {'RC_time':(rc,'s'),'carrier_time':(dt,'s'),'total_phase':(-math.degrees(math.atan(w*rc)+math.atan(w*dt)),'deg')},'tauRC=R*C; solve cascade amplitude; phases add with omega=2*pi*f', ['Changing cable C can move readout pole while intrinsic lifetime remains fixed.'])
    add('phase_magnitude_consistency','At 1000 Hz an alleged isolated first-order response has normalized magnitude 0.80 but measured phase -15 degrees. Infer time constants separately from magnitude and phase and their ratio. What conclusion is justified before a microscopic lifetime assignment?',
        {'tau_magnitude':(math.sqrt(1/.8**2-1)/(2*math.pi*1000),'s'),'tau_phase':(math.tan(math.radians(15))/(2*math.pi*1000),'s'),'ratio':(.75/math.tan(math.radians(15)),'1')},'tau_mag=sqrt(1/mag2-1)/omega; tau_phase=tan(absphase)/omega', ['Single-pole model is inconsistent; source timing, phase offset and multiple poles compete.'])
    b400=planck(9e-6,400);b300=planck(9e-6,300)
    add('Planck_differential_radiance',constants+'A 9 um narrow-band radiometer compares 400 K and 300 K ideal blackbodies. Compute each spectral radiance per um and their difference. Show the per-m to per-um conversion and explain why a temperature difference is not itself an optical-power calibration.',
        {'radiance_400K':(b400*1e-6,'W/(m^2*sr*um)'),'radiance_300K':(b300*1e-6,'W/(m^2*sr*um)'),'difference':((b400-b300)*1e-6,'W/(m^2*sr*um)')},'Planck SI B_lambda; multiply by 1e-6 for per-um radiance', ['Bandpass, projected throughput, emissivity and response weighting still required.'])
    integral=simpson(lambda lam:planck(lam,360),7.6e-6,10.3e-6);omega=math.pi*math.sin(math.radians(12))**2;power=integral*3e-8*omega*.7
    add('finite_band_projected_throughput','An ideal 360 K blackbody fills a circular cone of half-angle 12 degrees about a planar detector normal. Area is 3e-8 m^2; transmission is 0.70 over 7.6..10.3 um and zero elsewhere. Integrate Planck radiance to find band radiance, accepted projected solid angle and received power. Use int_cone cos(theta)dOmega=pi*sin(theta_max)^2.',
        {'band_radiance':(integral,'W/(m^2*sr)'),'projected_solid_angle':(omega,'sr'),'received_power':(power,'W')},'Composite Simpson Planck integral; Omega_projected=pi*sin2(theta); P=A*t*Omega*band_radiance', ['Projected solid angle already contains cosine; no second pi factor; uniform source radiance and flat transmission assumed.'])
    omega=math.pi/(4*2**2+1)
    add('finite_Fnumber_cone','An ideal geometrical cone has tan(theta_max)=1/(2F), F=2. Find exact projected solid angle and fractional overestimate from the paraxial pi/(4F^2) approximation. No vignetting; detector is planar on-axis. Explain which real optical effects are omitted.',
        {'projected_solid_angle':(omega,'sr'),'paraxial_relative_overestimate':((math.pi/16)/omega-1,'1')},'sin2(arctan(1/(2F)))=1/(4F2+1); Omega=pi*sin2', ['Cone definition supplied; actual pupil/etendue and vignetting may differ.'])
    observed=.7*planck(10e-6,340)+.3*planck(10e-6,295);lo,hi=295,340
    for _ in range(60):
        mid=(lo+hi)/2
        if planck(10e-6,mid)<observed:lo=mid
        else:hi=mid
    add('greybody_reflected_background','A 340 K opaque grey surface has spectral emissivity 0.70 at 10 um and reflects a uniform 295 K blackbody environment. Find observed spectral radiance per um and its single-wavelength equivalent blackbody temperature. Bound the equivalent temperature and state why one wavelength cannot determine both emissivity and true temperature.',
        {'observed_radiance':(observed*1e-6,'W/(m^2*sr*um)'),'equivalent_temperature':((lo+hi)/2,'K')},'Bobs=epsilon*B(Tobject)+(1-epsilon)*B(Tenvironment); monotonic bracketed Planck inversion', ['Positive convex mixture must lie between component radiances/temperatures; no temperature above both components.'])
    photons=simpson(lambda lam:planck(lam,390)*lam/(H*C),3.4e-6,4.6e-6);count=photons*5e-8*.02*.4
    add('band_photon_radiometry','A 390 K ideal blackbody has flat optical transmission 0.40 from 3.4 to 4.6 um. Detector area 5e-8 m^2 and projected solid angle 0.020 sr. Neglect cold-reference emission. Find incident photons/s in the band and diode current for wavelength-independent external QE 0.65. Explain why photon and energy integrals have different weights.',
        {'photon_rate':(count,'s^-1'),'photocurrent':(.65*Q*count,'A')},'Photon spectral radiance=B_lambda*lambda/(hc); integrate in metres; I=q*eta*photon_rate', ['Transmission and QE are separate; projected geometry and uniform band assumptions stated.'])
    alpha=700;th=.0015;lc=.002;eqe=.8*alpha/(alpha+1/lc)*(1-math.exp(-(alpha+1/lc)*th))
    add('absorption_collection_depth','At one wavelength a 15 um slab has alpha=700 cm^-1, front reflection 0.20 and no back reflection. A pair generated at depth z has collection probability exp(-z/Lc), Lc=20 um. Integrate external QE=(1-R)*int0t alpha*exp(-alpha*z)*exp(-z/Lc) dz. Also find absorbed fraction after front reflection, and distinguish them.',
        {'external_QE':(eqe,'1'),'absorbed_incident_fraction':(.8*(1-math.exp(-alpha*th)),'1')},'All depths converted to cm; EQE=(1-R)*alpha/(alpha+1/Lc)*(1-exp(-(alpha+1/Lc)*t))', ['Absorption is not collection; exponential collection model is supplied, not a universal device law.'])
    alpha=-math.log(.28/.55)/(.0025-.0010)
    add('transmission_thickness_inverse','Matched homogeneous slabs of thickness 10 and 25 um have single-pass power transmittances 0.55 and 0.28. Model T=C*exp(-alpha*t), common unknown reflection factor C; neglect interference and back reflections. Infer alpha in cm^-1 and C, and explain why mismatched surface finishes break the cancellation.',
        {'absorption_coefficient':(alpha,'cm^-1'),'surface_factor':(.55*math.exp(alpha*.001),'1')},'ln(T1/T2)=alpha*(t2-t1), cm thickness; C=T1*exp(alpha*t1)', ['Reflection factor cancels only if common; no claim that C determines refractive index uniquely.'])
    eta=.82*HC_EV_UM/1.55
    add('diode_EQE_calibration','A gain-free diode at 1.55 um has externally calibrated responsivity 0.82 A/W relative to incident power. Compute external QE and photocurrent for 0.30 uW. Would using absorbed power instead define the same QE?',
        {'external_QE':(eta,'1'),'photocurrent':(.82*.30e-6,'A')},'eta_ext=Ri*hc/(q*lambda); I=Ri*Pincident', ['External versus internal QE and electrical gain assumptions explicit.'])
    add('gain_responsivity_not_QE','A photoconductor at 5 um has current responsivity 14 A/W and independently estimated electrical gain 6. Infer external generation efficiency and the apparent QE one would infer if gain were ignored. Explain why apparent QE above unity is not evidence of energy creation.',
        {'generation_efficiency':(14*HC_EV_UM/5/6,'1'),'apparent_QE_without_gain':(14*HC_EV_UM/5,'1')},'Ri=eta*g*q*lambda/(hc); apparent EQE=eta*g', ['Bias supplies electrical gain energy; efficiency and gain estimates remain model dependent.'])
    rel=math.sqrt(.06**2+.04**2)
    add('NEP_calibration_uncertainty','Independent 1-sigma calibration uncertainties: noise ASD 6%, responsivity 4%. Nominal NEP density is 2 pW/sqrt(Hz). Find relative and absolute NEP uncertainty and D* relative uncertainty if active area also has independent 8% uncertainty. Use first-order propagation.',
        {'relative_NEP_sigma':(rel,'1'),'NEP_sigma':(2e-12*rel,'W/sqrt(Hz)'),'relative_Dstar_sigma':(math.sqrt(rel**2+(.08/2)**2),'1')},'log NEP=e-R; log Dstar=.5*logA-logNEP; independent fractional variances sum', ['Calibration covariance and nonlinear/model uncertainty omitted.'])
    add('Hall_density_uncertainty','Using single-carrier n=rH*I*B/(q*abs(VH)*t), independent relative 1-sigma errors are rH 5%, I 1%, B 2%, VH 3%, t 4%. Nominal n=2e21 m^-3. Find relative and absolute density uncertainty. Identify what a two-carrier model would change.',
        {'relative_sigma':(math.sqrt(.0055),'1'),'density_sigma':(2e21*math.sqrt(.0055),'m^-3')},'Independent logarithmic derivative variance=.05²+.01²+.02²+.03²+.04²=.0055', ['Statistics and Hall factor are model assumptions; precision is not validation of a single-carrier model.'])
    ea=.045;ratio=(110/80)**2*math.exp(-ea*Q/KB*(1/110-1/80))
    add('Arrhenius_prefactor','Dark current follows I=C*T^2*exp(-Ea/(kB*T)) over a limited interval. At 80 K I=2.0e-9 A; at 110 K I='+f'{2e-9*ratio:.10e}'+' A. Infer activation energy in eV using the supplied T^2 prefactor. Is this sufficient to identify diffusion, SRH or tunneling uniquely?',
        {'activation_energy':(ea,'eV'),'current_ratio':(ratio,'1')},'Ea=kB*ln[(I2/T2²)/(I1/T1²)]/(1/T1-1/T2), convert J to eV', ['Apparent Arrhenius energy/prefactor over two points is not a unique mechanism assignment.'])
    add('contact_geometry_diagnosis','Same-cross-section devices have low-field two-terminal R=140 Ohm at L=200 um and 260 Ohm at L=500 um. Cross-section 8e-10 m^2. Model R=2Rc+rho*L/S with common equal ohmic contacts. Infer rho and per-contact Rc. After passivation both resistances decrease by 20 Ohm with the same slope. Find new Rc; give competing explanations and a four-terminal control.',
        {'bulk_resistivity':(120/300e-6*8e-10,'Ohm*m'),'contact_resistance_before':(30,'Ohm'),'contact_resistance_after':(20,'Ohm')},'Geometry slope=rho/S; intercept=2Rc; common shift changes intercept under model', ['Contact inference conditional on common geometry/material; parallel surface leakage/current crowding can violate linear model.'])
    add('readout_calibration_artifact','An apparent detector response rises from 10 to 20 mV for unchanged incident modulation. A dummy-resistor injected-current control rises by the same factor; independently calibrated detector current remains 5 uA. Find transimpedance before/after and detector-current ratio. What claim about passivation-enhanced carrier gain is supported?',
        {'transimpedance_before':(.010/5e-6,'Ohm'),'transimpedance_after':(.020/5e-6,'Ohm'),'current_ratio':(1,'1')},'Z=V/I; same current with doubled readout gain explains doubled voltage', ['Readout change suffices; no carrier-gain improvement follows without independent optical/current and noise controls.'])
    density=1.5e21;mu=.6;l=250e-6;section=75e-6*6e-6;bias=.10;life=2e-6
    resistance=l/(Q*density*mu*section);transit=l*l/(mu*bias);gain=life/transit;ri=.45*gain*Q*6e-6/(H*C);si=4*KB*90/resistance+(4e-12)**2;nep=math.sqrt(si)/ri
    add('linked_detector_design',constants+'An ideal ohmic single-carrier photoconductor has n=1.5e21 m^-3, drift mobility 0.60 m^2/(V*s), length 250 um, width 75 um, thickness 6 um, active bias 0.10 V, independent effective lifetime 2 us, T=90 K. At 6 um external generation efficiency is 0.45. Assume low-field gain tau/tau_transit, white one-sided Johnson current noise plus independent 4 pA/sqrt(Hz) readout current noise. Optical active area is LENGTH*WIDTH. Compute resistance, transit time, gain, current responsivity, NEP density, D* in Jones, detector-only f3dB and Joule power. Distinguish this ideal prediction from a measured design claim.',
        {'resistance':(resistance,'Ohm'),'transit_time':(transit,'s'),'gain':(gain,'1'),'responsivity':(ri,'A/W'),'nep_density':(nep,'W/sqrt(Hz)'),'dstar':(math.sqrt(l*75e-6*1e4)/nep,'Jones'),'f3dB':(1/(2*math.pi*life),'Hz'),'joule_power':(bias*bias/resistance,'W')},
        'sigma=q*n*mu; R=L/(sigma*Wt); transit=L2/(muV); g=tau/transit; Ri=eta*g*q*lambda/(hc); SI=4kBT/R+readout2; Dstar=sqrt(A_cm2)/NEP',
        ['Lifetime independent; optical area differs from electrical cross-section; low-field gain/contact/bias conventions explicit.', 'Shot/GR/1f/background omitted, so this is a restricted ideal noise budget; do not claim achieved sensitivity.'])
    assert len(items)==40
    return items


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('evals/firm_science_dev_v1.jsonl'));args=ap.parse_args()
    manifest=args.out.with_name(args.out.stem+'_manifest.json')
    if args.out.exists() or manifest.exists():ap.error('Release exists; use a new version')
    items=development_items()
    from eval_firm_science import validate_benchmark
    validate_benchmark(items)
    write_jsonl(args.out,items)
    write_json(manifest,{'schema_version':'1.0','benchmark_version':'firm_science_dev_v1','generation_date':'2026-10-05',
        'status':'public development suite; analytic AI review and oracle tests; human expert signoff pending',
        'answers_public':True,'source_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p):sha256(p) for p in [Path(__file__),Path('scripts/firm_science_reference.py')]},
        'cases':len(items),'quantities':sum(len(x['grading']['quantities']) for x in items),
        'assets':[{'path':str(args.out),'sha256':sha256(args.out),'records':[{'id':x['id'],'record_sha256':digest(x),'category':x['category'],'family_id':x['family_id']} for x in items]}],
        'leakage_policy':'All recognized legacy quantitative families reserved from reviewed seed; no seed text used by generator. Lexical and manual release checks still required.',
        'limitations':'Not a hidden or permanent unbiased benchmark; shared core equations are expected domain knowledge.'})
    print('cases',len(items),'quantities',sum(len(x['grading']['quantities']) for x in items))


if __name__=='__main__':main()
