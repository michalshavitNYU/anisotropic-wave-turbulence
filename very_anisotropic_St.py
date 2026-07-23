"""Collision integral in the VERY ANISOTROPIC limit (k ~ rho, A2D = A3D exact,
linear resonance -> verticals in closed form). Spectrum e = rho^{-a} |k_z|^{-b}.
Claim to test: (a,b) = (5/2, 1/2), i.e. e = rho^{-5/2}|k_z|^{-1/2}
(= k^{-3}|cos theta|^{-1/2}), is a zero of St.

Fix rho_alpha = 1, k_alpha^z = 1, s_alpha = +1. Then k_i^z = mu_i with
  mu_beta = (s_g rho_g - 1)/D,  mu_gamma = (1 - s_b rho_b)/D,  D = s_b rho_b - s_g rho_g,
and per branch (s_b, s_g):
  St = int drho_b drho_g  A |D| (1 + s_b rho_b + s_g rho_g)^2 / (8 rho_b rho_g)
        * [ e_b e_g + mu_b e_g + mu_g e_b ],     e_i = rho_i^{-a} |mu_i|^{-b}
over the horizontal triangle |rho_b - 1| <= rho_g <= rho_b + 1, rho <= 10^h,
with the paper's box regulator 10^{-d} off the singular lines.
Equilibrium check: a=b=0 -> bracket = 1 + mu_b + mu_g = 0 identically."""
import numpy as np
from scipy.integrate import quad

def A2D(rb, rg):
    s = 4.0 * rb * rb * rg * rg - (rb * rb + rg * rg - 1.0) ** 2
    return 0.25 * np.sqrt(s) if s > 0.0 else 0.0

def integrand(rg, rb, sb, sg, a, b):
    D = sb * rb - sg * rg
    if D == 0.0:
        return 0.0
    A = A2D(rb, rg)
    if A <= 0.0:
        return 0.0
    mb = (sg * rg - 1.0) / D
    mg = (1.0 - sb * rb) / D
    if mb == 0.0 or mg == 0.0:
        return 0.0
    S = 1.0 + sb * rb + sg * rg
    eb = rb ** (-a) * abs(mb) ** (-b)
    eg = rg ** (-a) * abs(mg) ** (-b)
    br = eb * eg + mb * eg + mg * eb          # e_alpha = 1
    return A * abs(D) * S * S * br / (8.0 * rb * rg)

def St(a, b, h=2.0, d=6, branches=None, epsrel=1e-8):
    R = 10.0 ** h
    eps = 10.0 ** (-d)
    if branches is None:
        branches = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    total = 0.0
    for sb, sg in branches:
        def inner(rb):
            lo = abs(rb - 1.0) + eps
            hi = min(rb + 1.0, R) - eps
            if hi <= lo:
                return 0.0
            pts = [p for p in (1.0, rb) if lo < p < hi]
            v, _ = quad(integrand, lo, hi, args=(rb, sb, sg, a, b),
                        points=pts or None, limit=200, epsabs=1e-11, epsrel=epsrel)
            return v
        v, _ = quad(inner, eps, R, points=[1.0, 2.0], limit=300,
                    epsabs=1e-10, epsrel=epsrel)
        total += v
    return total

if __name__ == "__main__":
    print("equilibrium check  a=b=0 :", St(0.0, 0.0), "(must be ~0)", flush=True)
    print("non-solution check a=1,b=0:", St(1.0, 0.0), "(must be nonzero)", flush=True)
    print("\nb-scan at a=5/2  (h=2, d=6):", flush=True)
    for b in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        print(f"  b={b:.2f}  St={St(2.5, b):+.6e}", flush=True)
    print("\na-scan at b=1/2:", flush=True)
    for a in (2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 2.9):
        print(f"  a={a:.2f}  St={St(a, 0.5):+.6e}", flush=True)
    print("\nregulator convergence at (a,b)=(2.5,0.5):", flush=True)
    for h, d in ((1.0, 4), (2.0, 6), (3.0, 8)):
        print(f"  h={h} d={d}:  St={St(2.5, 0.5, h=h, d=d):+.6e}", flush=True)
