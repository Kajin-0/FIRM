"""Small SI reference functions for public development oracles, not a detector simulator.

All PSDs are one-sided in Hz unless explicitly stated. Geometry and physical
assumptions belong in each problem. Empirical HgCdTe coefficients are supplied
models, not universal material constants. No training answers are imported here.
"""
import math

Q = 1.602176634e-19
KB = 1.380649e-23
H = 6.62607015e-34
C = 299792458.0
HC_EV_UM = H * C / Q * 1e6


def gap(x, t):
    return -.302 + (1.93 - .810*x + .832*x*x)*x + .000535*t*(1-2*x)


def gap_dx(x, t):
    return 1.93 - 1.620*x + 2.496*x*x - .001070*t


def composition(cutoff_um, temperature):
    """Newton root, independently implemented from the v1 pilot's bisection."""
    target, x = HC_EV_UM/cutoff_um, .25
    for _ in range(30):
        x -= (gap(x, temperature)-target)/gap_dx(x, temperature)
    if not .15 < x < .5 or abs(gap(x, temperature)-target) > 1e-12:
        raise ValueError('Root outside the stated development bracket')
    return x


def planck(wavelength_m, temperature):
    """Spectral radiance in W/(m^2 sr m), SI wavelength, positive temperature."""
    if wavelength_m <= 0 or temperature <= 0:
        raise ValueError('Planck requires positive wavelength and temperature')
    exponent = H*C/(wavelength_m*KB*temperature)
    return 0.0 if exponent > 700 else 2*H*C*C/(wavelength_m**5*math.expm1(exponent))


def simpson(function, lo, hi, panels=8192):
    if panels < 2 or panels % 2 or hi <= lo:
        raise ValueError('Even positive panel count and increasing interval required')
    step = (hi-lo)/panels
    total = function(lo)+function(hi)
    total += math.fsum((4 if k % 2 else 2)*function(lo+k*step) for k in range(1, panels))
    return total*step/3


def enbw(time_constant, poles=1):
    """Integral 0..infinity of [1+(2*pi*f*tau)^2]^-poles, unity DC gain."""
    if time_constant <= 0 or poles < 1:
        raise ValueError('Positive time constant and positive pole count required')
    return math.gamma(poles-.5)/(4*math.sqrt(math.pi)*time_constant*math.gamma(poles))


def lorentzian_variance(s0, tau, lo=0, hi=math.inf):
    return s0/(2*math.pi*tau)*(math.atan(2*math.pi*tau*hi)-math.atan(2*math.pi*tau*lo))


def detectivity(area_m2, nep_rms_w, bandwidth_hz):
    if min(area_m2, nep_rms_w, bandwidth_hz) <= 0:
        raise ValueError('Positive area, NEP and bandwidth required')
    # Convert the AREA before taking its square root; this independently checks
    # the legacy error instead of blindly multiplying its printed answer.
    return math.sqrt(area_m2*1e4*bandwidth_hz)/nep_rms_w


def pilot_rederivation():
    """Independent substitutions and integration; never call the v1 generator."""
    x = composition(9.4, 85)
    amp, omega, rc = .63418238, 2*math.pi*1200, 40e-6
    # Read the supplied rounded amplitude, rather than constructing it from tau.
    tau = math.sqrt((1/amp**2)/(1+(omega*rc)**2)-1)/omega
    mu, density = 9000/1e4, 2e15*1e6
    resistance = 400e-6/(Q*density*mu*80e-6*12e-6)
    transit = 400e-6/((mu*.2)/(400e-6))
    variance = simpson(lambda logf: (4e-16/math.exp(logf)+9e-18)*math.exp(logf), math.log(2), math.log(500))
    # Decimal independently verifies the sensitive Planck exponential/conversion.
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 45
        h, c, kb, lam, t = map(Decimal, ['6.62607015e-34','299792458','1.380649e-23','0.000008','420'])
        b = float(2*h*c*c/(lam**5*((h*c/(lam*kb*t)).exp()-1)))
    power = b*.080e-6*2e-8*.015*.55
    return [
        {'gap_85K': HC_EV_UM/9.4, 'composition_x': x, 'cutoff_150K': HC_EV_UM/gap(x,150)},
        {'nep_before': 25e-9/1e4, 'nep_after': 6e-9/7e3,
         'dstar_before': math.sqrt(.4/100)/(25e-9/1e4), 'dstar_after': math.sqrt(.4/100)/(6e-9/7e3),
         'dstar_ratio': (25e-9/1e4)/(6e-9/7e3)},
        {'carrier_tau': tau, 'detector_f3db': 1/(omega*tau)*1200},
        {'dark_resistance': resistance, 'transit_time': transit, 'gain': 3e-6/transit},
        {'corner_frequency': 4e-16/9e-18, 'rms_noise': math.sqrt(variance)},
        {'radiance_per_um': b/1e6, 'modulated_power_difference': power, 'voltage_difference': power*8000},
        {'thickness_95pct': math.log(20)*10, 'absorption_20um': -math.expm1(-2)},
        {'enbw': enbw(.020), 'rms_noise': 12e-9*math.sqrt(enbw(.020))}]
