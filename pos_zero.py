"""Locate the zero of the positive-branch collision integral St^+_exact(k_z).

St^+ (s_alpha=s_beta=s_gamma=+1) is a small residual after channel cancellation,
so it needs HIGH azimuthal resolution (nph>=160); nph=64 gives the wrong sign.
The zero LOCATION is far more stable than the magnitudes around it.

Result (odd-wave, a=1/2):  k_z* ~ 0.32-0.33, roughly regulator-independent
  eps=1e-3  -> k_z* ~ 0.33
  eps=3e-4  -> k_z* ~ 0.32
"""
import numpy as np
from collision_integral import St
from scipy.optimize import brentq

POS = [(+1.0, +1.0)]

def st_plus(z, eps=1e-3, nph=192):
    Tm = float(np.arccosh(2.0 / eps)) + 4.0
    return St(z, nt=130, nph=nph, Tmax=Tm, eps=eps, f=0.5, branches=POS)[0]

if __name__ == "__main__":
    eps = 1e-3
    zs = np.round(np.arange(0.22, 0.40, 0.02), 3)
    vals = [st_plus(z, eps) for z in zs]
    for z, v in zip(zs, vals):
        print(f"  k_z={z:.3f}  St+={v:+.4e}", flush=True)
    for i in range(len(zs) - 1):
        if vals[i] * vals[i + 1] < 0:
            zc = brentq(lambda z: st_plus(z, eps), zs[i], zs[i + 1], xtol=3e-4)
            print(f"  zero crossing:  k_z* = {zc:.4f}", flush=True)
