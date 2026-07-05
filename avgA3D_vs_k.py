"""Partial average of the 3D triangle area over the two "leg" wavevectors.

Computes  <A3D>_{beta,gamma}(k_alpha)  =  int dV_beta dV_gamma  A3D  e_beta e_gamma
                                                 * delta(k_alpha + k_beta + k_gamma),
i.e. A3D averaged over p = k_beta and q = k_gamma against the distribution
e_p e_q delta(p + k + q). By homogeneity this is a function of the alpha leg only;
we fix |k_alpha| = 1 and evaluate it as a function of k_alpha^z = cos(theta).

Areas (Heron), with the corrected A3D (note the -kaz**2 term, which the earlier
Julia code and the manuscript appendix were missing):
    A2D = (rho_alpha**2 / 4) sqrt[ 4(u+v)**2 (u-v)**2 - (2u**2 + 2v**2 - 1)**2 ]
    A3D = (1/4) sqrt[ 4 k_beta**2 k_gamma**2 - (k_beta**2 + k_gamma**2 - k_alpha**2)**2 ],
          k_alpha**2 = rho_alpha**2 + kaz**2  (so the bracket carries -kaz**2).

Kinematic-box variables (rho_beta = rho_alpha (u+v), rho_gamma = rho_alpha (u-v),
k_gamma^z = -(kaz + kbz)); the azimuthal delta functions give I_phi = 1/(2 A2D) and
the (rho_beta, rho_gamma) -> (u, v) Jacobian gives 2 rho_alpha**4 (u+v)(u-v), so
    <A3D>_{beta gamma} = rho_alpha**4 int du dv dkbz (u+v)(u-v) (A3D/A2D) e_beta e_gamma.

Python port of avgA3D_vs_k.jl. Created by Shahaf Aharony (Weizmann Institute) and
Michal Shavit; area correction and swappable-spectrum generalization added here.
"""

from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import integrate


# ----------------------------------------------------------------------------------
# Spectrum e(k), written as a function of (rho**2, kz). Swap this single function to
# compare different probability distributions P ~ e_p e_q delta(p + k + q).
#   Odd-wave spectrum:  e = |kz|**(-1/2) (rho**2 + kz**2)**(-5/4)  ==  k**-3 |cos th|**-1/2
# ----------------------------------------------------------------------------------
def espec_odd(rho2, kz):
    return abs(kz) ** (-0.5) * (rho2 + kz * kz) ** (-1.25)


def espec_iso(rho2, kz):          # e = 1 (thermal / isotropic)
    return 1.0


def espec_pow(rho2, kz, s=1.25):  # isotropic power law
    return (rho2 + kz * kz) ** (-s)


def espec_gen(rho2, kz, s=1.25, a=0.5):  # tunable anisotropy
    return abs(kz) ** (-a) * (rho2 + kz * kz) ** (-s)


# ----------------------------------------------------------------------------------
# Areas in the (u, v, kbz) kinematic box, at fixed alpha leg with |k_alpha| = 1.
# ----------------------------------------------------------------------------------
def a2d_sqrt(rhoA2, u, v):
    return 4 * (u + v) ** 2 * (u - v) ** 2 - (2 * u * u + 2 * v * v - 1) ** 2


def a3d_sqrt(rhoA2, kaz, kbz, u, v):
    return (
        4 * (rhoA2 * (u + v) ** 2 + kbz * kbz) * (rhoA2 * (u - v) ** 2 + (kaz + kbz) ** 2)
        - (rhoA2 * (2 * u * u + 2 * v * v - 1) - kaz * kaz + kbz * kbz + (kaz + kbz) ** 2) ** 2
    )


def integrand_a3d(u, v, kbz, kaz, e):
    """Integrand of <A3D>_{beta gamma} in (u, v, kbz); spectrum e is swappable."""
    rhoA2 = 1.0 - kaz * kaz              # rho_alpha**2, since |k_alpha| = 1
    a2s = a2d_sqrt(rhoA2, u, v)
    a3s = a3d_sqrt(rhoA2, kaz, kbz, u, v)
    if a2s <= 0.0 or a3s <= 0.0:
        return 0.0
    A2D = rhoA2 * np.sqrt(a2s) / 4.0
    A3D = np.sqrt(a3s) / 4.0
    eb = e(rhoA2 * (u + v) ** 2, kbz)          # e_beta  (rho_beta**2 = rhoA2 (u+v)**2)
    eg = e(rhoA2 * (u - v) ** 2, kaz + kbz)    # e_gamma (k_gamma^z = -(kaz + kbz))
    return rhoA2 * rhoA2 * (u + v) * (u - v) * (A3D / A2D) * eb * eg


def integrand_a2d(u, v, kbz, kaz, e):
    """<A2D>_{beta gamma}: same measure without A3D/A2D (the A2D cancels 1/A2D)."""
    rhoA2 = 1.0 - kaz * kaz
    a2s = a2d_sqrt(rhoA2, u, v)
    if a2s <= 0.0:
        return 0.0
    eb = e(rhoA2 * (u + v) ** 2, kbz)
    eg = e(rhoA2 * (u - v) ** 2, kaz + kbz)
    return rhoA2 * rhoA2 * (u + v) * (u - v) * eb * eg


# ----------------------------------------------------------------------------------
# Numerical integration over (u, v, kbz) with the kinematic-box regulators (h, d):
# u in [1/2 + 10**-d, 10**h], v in [-1/2 + 10**-d, 1/2 - 10**-d], and kbz split around
# its two branch-point singularities kbz = 0 and kbz = -kaz.
# ----------------------------------------------------------------------------------
def kbz_segments(kaz, h, d):
    reg = 10.0 ** (-d)
    top = 10.0 ** h
    if kaz < 0:
        return [(-top, -reg), (reg, -kaz - reg), (-kaz + reg, top)]
    return [(-top, -kaz - reg), (-kaz + reg, -reg), (reg, top)]


def avg_bg(integrand, kaz, h=6, d=5, atol=1e-4, e=espec_odd, limit=100):
    reg = 10.0 ** (-d)
    u_range = [0.5 + reg, 10.0 ** h]
    v_range = [-0.5 + reg, 0.5 - reg]
    opts = {"epsabs": atol, "epsrel": 1e-4, "limit": limit}
    val = 0.0
    err2 = 0.0
    for klo, khi in kbz_segments(kaz, h, d):
        # nquad order: x0=u (innermost), x1=v, x2=kbz.
        res, err = integrate.nquad(
            integrand,
            [u_range, v_range, [klo, khi]],
            args=(kaz, e),
            opts=[opts, opts, opts],
        )
        val += res
        err2 += err * err
    return val, np.sqrt(err2)


def avg_a3d_bg(kaz, h=6, d=5, atol=1e-4, e=espec_odd):
    return avg_bg(integrand_a3d, kaz, h, d, atol, e)


def avg_a2d_bg(kaz, h=6, d=5, atol=1e-4, e=espec_odd):
    return avg_bg(integrand_a2d, kaz, h, d, atol, e)


def _a3d_worker(kaz, h, d, atol, e):
    return avg_a3d_bg(kaz, h, d, atol, e)


def a3d_vs_k(kaz_grid, h=6, d=5, atol=1e-4, e=espec_odd, processes=None):
    """<A3D>_{beta gamma} on a grid of k_alpha^z, parallelized over the grid."""
    work = partial(_a3d_worker, h=h, d=d, atol=atol, e=e)
    with Pool(processes=processes) as pool:
        out = pool.map(work, kaz_grid)
    vals = np.array([o[0] for o in out])
    errs = np.array([o[1] for o in out])
    return vals, errs


if __name__ == "__main__":
    spectrum = espec_odd                 # <-- swap to compare distributions
    kaz_grid = np.arange(0.001, 1.0, 0.01)

    vals, errs = a3d_vs_k(kaz_grid, h=6, d=5, atol=1e-4, e=spectrum)

    # Optional: save / plot
    # np.savetxt("avgA3D_vs_k.csv",
    #            np.column_stack([kaz_grid, vals, errs]),
    #            delimiter=",", header="kaz,A3D,A3D_err", comments="")
    # import matplotlib.pyplot as plt
    # plt.plot(kaz_grid, vals); plt.xlabel(r"$k_\alpha^z$")
    # plt.ylabel(r"$\langle A_{3D}\rangle_{\beta\gamma}$"); plt.show()

    for k, a, e in zip(kaz_grid[::10], vals[::10], errs[::10]):
        print(f"kaz={k:.3f}   <A3D>_bg={a:.6g}   err={e:.2g}")
