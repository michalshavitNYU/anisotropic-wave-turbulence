"""Galerkin correction of the angular profile: e = k^{-3}|cos|^{-1/2} G(theta),
G = sum_n a_n cos(2n theta), a_0 = 1. St is bilinear in e, so
St(z; a) = sum_{mn} a_m a_n M_mn(z); compute M once, then optimize a cheaply."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from collision_integral import _roots, _omega_deriv

W, F, EPS, SA = 3.0, 0.5, 1e-3, +1.0
Tm = float(np.arccosh(2.0 / EPS)) + 4.0
NT, NPH = int(7 * Tm), 80

def base_e(kz, k):
    return k ** (-W + F) * (kz * kz + EPS * EPS) ** (-F / 2)

def gbasis(c, M):
    th = np.arccos(np.clip(c, -1.0, 1.0))
    return np.array([np.cos(2 * n * th) for n in range(M + 1)])

def sym(u, v):
    return 0.5 * (np.outer(u, v) + np.outer(v, u))

def St_matrix(z, M):
    rhoA2 = 1.0 - z * z; rhoA = np.sqrt(rhoA2)
    Ea = base_e(z, 1.0); ga = gbasis(z, M)
    xt, wt = leggauss(NT); t = 0.5 * Tm * (xt + 1); wt = 0.5 * Tm * wt
    xp, wp = leggauss(NPH); ph = 0.5 * np.pi * xp; wp = 0.5 * np.pi * wp
    Mmat = np.zeros((M + 1, M + 1))
    for i in range(NT):
        u = 0.5 * np.cosh(t[i])
        for j in range(NPH):
            v = 0.5 * np.sin(ph[j])
            rb = rhoA * (u + v); rg = rhoA * (u - v)
            meas = 2.0 * rhoA2 ** 2 * (u + v) * (u - v)
            Wij = wt[i] * wp[j]
            Xm = 8.0 + 6.0 * max(rb, rg)
            for sb in (+1.0, -1.0):
                for sg in (+1.0, -1.0):
                    for x in _roots(z, rb, rg, sb, sg, Xm):
                        kb = np.hypot(rb, x); kgz = -(z + x); kg = np.hypot(rg, kgz)
                        a3s = 4 * kb ** 2 * kg ** 2 - (kb ** 2 + kg ** 2 - 1.0) ** 2
                        if a3s <= 0:
                            continue
                        dOm = abs(_omega_deriv(x, z, rb, rg, sb, sg))
                        if dOm < 1e-14:
                            continue
                        A3D2 = a3s / 16.0
                        G2 = (A3D2 / (4 * z * z * kb ** 2 * kg ** 2)
                              * (sb * kb - sg * kg) ** 2 * (SA + sb * kb + sg * kg) ** 2)
                        Eb = base_e(x, kb); Eg = base_e(kgz, kg)
                        gb = gbasis(x / kb, M); gg = gbasis(kgz / kg, M)
                        Br = (z * Eb * Eg * sym(gb, gg)
                              + x * Ea * Eg * sym(ga, gg)
                              + kgz * Ea * Eb * sym(ga, gb))
                        Mmat += (Wij * meas / (2 * rhoA2)) * (z * G2 / dOm) * Br
    return Mmat

if __name__ == "__main__":
    M = 2
    zs = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    mats = []
    for z in zs:
        Mm = St_matrix(z, M)
        mats.append(Mm)
        print(f"z={z}: M00={Mm[0,0]:+.4e} (St of pure |cos|^-1/2)", flush=True)
    np.save("galerkin_mats.npy", np.array(mats))
    print("saved galerkin_mats.npy", flush=True)

    from scipy.optimize import minimize
    def resid(c):
        a = np.concatenate(([1.0], c))
        return sum((a @ Mm @ a) ** 2 for Mm in mats)
    base = resid([])
    r = minimize(lambda c: resid(c), [0.0, 0.0], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-14, maxiter=2000))
    a = np.concatenate(([1.0], r.x))
    print(f"\nbaseline  sum St^2 = {base:.4e}", flush=True)
    print(f"optimized sum St^2 = {r.fun:.4e}   (reduction x{base/r.fun:.1f})", flush=True)
    print(f"coefficients: a = {a}", flush=True)
    for z, Mm in zip(zs, mats):
        print(f"  z={z}:  St_base={Mm[0,0]:+.4e}   St_opt={a @ Mm @ a:+.4e}", flush=True)
    th = np.linspace(0, np.pi / 2, 7)
    G = sum(a[n] * np.cos(2 * n * th) for n in range(M + 1))
    print("G(theta) at theta/pi =", np.round(th / np.pi, 3), ":", np.round(G, 4), flush=True)
