"""
Convergence of the regulated (h, d) box integral to the exact ratio.

The manuscript regulates the (u, v, k_beta^z) box as
    u in [1/2 + 10^-d, 10^h],   v in [-1/2 + 10^-d, 1/2 - 10^-d],
    k_beta^z: within 10^-d of the branch points 0 and -k_alpha^z removed, |k_beta^z| <= 10^h.
Here we evaluate the SAME regulated box but in the smooth substitution variables of
ratio_quadrature.py (so each regulated value is itself accurate), by simply truncating
the integration limits. As (h, d) -> infinity the limits fill the full range and the
result converges to the exact ratio-of-averages computed in ratio_quadrature.py.

Produces the data for the ratio(h) and ratio(d) convergence figure.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def _gl(n, lo, hi):
    x, w = leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def partial_N_reg(z, a, h, d, nt=90, nph=70, nkb=90):
    """N2, N3 on the regulated box [10^-d ... 10^h], in smooth variables."""
    eps = 10.0 ** (-d)
    Umax = 10.0 ** h
    rhoA2 = 1.0 - z * z

    # u = 1/2 cosh t in [1/2+eps, Umax]  ->  t in [tlo, thi]
    tlo = np.arccosh(1.0 + 2.0 * eps)
    thi = np.arccosh(2.0 * Umax)
    # v = 1/2 sin(phi) in [-1/2+eps, 1/2-eps] -> phi in [-phc, phc]
    phc = np.arcsin(1.0 - 2.0 * eps)

    t, wt = _gl(nt, tlo, thi)
    ph, wph = _gl(nph, -phc, phc)
    u = 0.5 * np.cosh(t); v = 0.5 * np.sin(ph)
    u2 = u * u; v2 = v * v
    U2V2 = u2[:, None] - v2[None, :]
    UPV = u[:, None] + v[None, :]
    UMV = u[:, None] - v[None, :]
    SH = np.sinh(t)[:, None] + 0.0 * ph[None, :]
    CO = np.cos(ph)[None, :] + 0.0 * u[:, None]
    P2 = 2 * u2[:, None] + 2 * v2[None, :] - 1.0
    wtph = wt[:, None] * wph[None, :]

    N2 = 0.0; N3 = 0.0
    for seg in ("M", "R", "L"):
        if seg == "M":
            # k_b^z in [-z+eps, -eps]  (branch cutoffs); empty if z <= 2 eps
            if z <= 2 * eps:
                continue
            xhi = np.arcsin(min(1.0, 1.0 - 2 * eps / z))
            xlo = np.arcsin(max(-1.0, -1.0 + 2 * eps / z))
            xi, wk = _gl(nkb, xlo, xhi)
            c = np.cos(xi)
            kb = -z / 2 + (z / 2) * np.sin(xi)
            SJ = wk * (z / 2) ** (1 - 2 * a) * c ** (1 - 2 * a)
            brf = 1.0
        else:
            # k_b^z = s^2 (R) or -z - s^2 (L); branch cut s>=sqrt(eps), outer |k_b^z|<=Umax.
            # Integrate in the bounded variable r = s/(1+s) so wide s-ranges stay resolved.
            slo = np.sqrt(eps)
            shi = np.sqrt(Umax) if seg == "R" else np.sqrt(max(Umax - z, eps))
            rlo = slo / (1.0 + slo); rhi = shi / (1.0 + shi)
            r, wr = _gl(nkb, rlo, rhi)
            s = r / (1.0 - r); dsdr = 1.0 / (1.0 - r) ** 2
            SJ = wr * 2.0 * s ** (1 - 2 * a) * dsdr
            if seg == "R":
                kb = s * s
                brf = np.abs(z + kb) ** (-a)
            else:
                kb = -z - s * s
                brf = np.abs(z + s * s) ** (-a)
        kb = kb[None, None, :]; kg = z + kb
        upv = UPV[:, :, None]; umv = UMV[:, :, None]
        beta2 = rhoA2 * upv ** 2 + kb ** 2
        gam2 = rhoA2 * umv ** 2 + kg ** 2
        W = beta2 ** (-(3 - a) / 2) * gam2 ** (-(3 - a) / 2) * brf
        X = rhoA2 * P2[:, :, None] - z * z + kb ** 2 + kg ** 2
        A3D = 0.25 * np.sqrt(np.clip(4.0 * beta2 * gam2 - X ** 2, 0.0, None))
        base = wtph[:, :, None] * SJ[None, None, :] * W
        N2 += rhoA2 ** 2 * 0.25 * np.sum(base * U2V2[:, :, None] * SH[:, :, None] * CO[:, :, None])
        N3 += rhoA2 * np.sum(base * U2V2[:, :, None] * A3D)
    return N2, N3


# Fixed high-order outer angular rule: A = int N3 z^-a dz / int N2 z^-a dz,
# with z = w^{1/(1-a)} so z^-a dz = 1/(1-a) dw (uniform in w).
def _outer_nodes(a, nz=90):
    w, ww = _gl(nz, 0.0, 1.0)
    z = w ** (1.0 / (1.0 - a))
    return z, ww / (1.0 - a)          # weights already fold in z^-a dz


def ratio_of_averages_reg(a, h, d, nz=90):
    z, wz = _outer_nodes(a, nz)
    num = den = 0.0
    for zi, wi in zip(z, wz):
        n2, n3 = partial_N_reg(zi, a, h, d)
        num += wi * n3; den += wi * n2
    return num / den


if __name__ == "__main__":
    a = 0.5
    EXACT = 1.311037
    print(f"Regulator convergence of ratio-of-averages, a={a} (exact = {EXACT}):\n")
    print("vary h at fixed d=6:")
    for h in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        R = ratio_of_averages_reg(a, h, 6)
        print(f"  h={h:3.1f}  d=6 :  {R:.5f}   (err {R-EXACT:+.2e})")
    print("\nvary d at fixed h=3:")
    for d in [1, 2, 3, 4, 5, 6]:
        R = ratio_of_averages_reg(a, 3.0, d)
        print(f"  h=3  d={d} :  {R:.5f}   (err {R-EXACT:+.2e})")
