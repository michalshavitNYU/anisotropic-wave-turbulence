"""
Full collision integral St_alpha(k_alpha^z) at FINITE anisotropy (full k = sqrt(rho^2+kz^2)),
with the frequency resonance resolved on the manifold. Compares
    St_exact  : keeps the exact azimuthal Jacobian I_phi = 1/(2 A2D)
    St_approx : the paper's approximation  1/(2 A2D) -> 1/(2 A3D)
so that (St_exact - St_approx)/St_exact is the true error of <A2D> ~ <A3D>.

Odd waves:  omega = sigma * k_z * k   (nu=1).   Spectrum  e = k^{-w+f} |k_z|^{-f},  (w,f)=(3,1/2).
Vertex (Waleffe):  |Gamma|^2 = A3D^2 /(4 k_az^2 k_a^2 k_b^2 k_g^2)
                                * (sb k_b - sg k_g)^2 (sa k_a + sb k_b + sg k_g)^2 .
Kinetic eq:  St_a = sum_{sb,sg} int d(u) d(v) [meas] sum_roots  k_az |Gamma|^2 S_spec
                    / (2 A_ND) / |dOmega/dk_bz| ,
   meas = 2 rho_a^4 (u+v)(u-v),   S_spec = k_az e_b e_g + k_bz e_a e_g + k_gz e_a e_b,
   rho_b = rho_a(u+v), rho_g = rho_a(u-v),  k_gz = -(k_az + k_bz),  |k_a|=1.

FINDING (2026-07): the resonant manifold is resolved correctly (roots found per
configuration; the (u,v) angular integration is node-converged to <0.1%). HOWEVER the
raw St_alpha at FINITE anisotropy does NOT converge in the leg-size (large-u) integral:
as the u-cutoff grows (Tmax 5->11, u up to ~1e4) St drifts monotonically and even
CHANGES SIGN (e.g. k_az=0.05: -121 -> -57 -> -15 -> +17). The large-|k_beta|,|k_gamma|
(hard-leg) contributions never die off -- a marginal-locality / slow-mode divergence.

Consequence: raw St_alpha (and hence a raw St_exact/St_approx comparison) is NOT a
well-defined convergent observable at finite anisotropy. It must be regularized (the
paper's h,d) AND the physically meaningful, SATURATING quantity is the radial FLUX
Pi (the paper's Fig. 3), not the bare collision integral. Note the entanglement: the
very-anisotropic limit k~rho that makes St saturate is exactly where A3D=A2D (zero
error), so the error lives in the part that must be regularized.

RESOLUTION (slow-mode regulator eps): the divergence is a slow-mode effect -- at large
u the resonant root has |k_beta^z| ~ 1/u -> 0. Regularizing |k_z|^{-1/2} ->
(k_z^2+eps^2)^{-1/4} on all three legs makes each St(eps) converge in the leg-size
integral. As eps -> 0:
  * St_exact(eps) and St_approx(eps) both DIVERGE logarithmically (slow modes), but
  * their DIFFERENCE St_exact - St_approx converges to a finite, eps-independent value
    (the divergent slow-mode part is common, since A3D=A2D there).
So the approximation error is a genuine finite quantity (no divergence introduced). The
error profile St_exact - St_approx (eps-converged) vs k_alpha^z:
    k_az :  0.10   0.20   0.30   0.50   0.70
    err  : +1.35  +1.14  +0.59  -0.19  -0.57   (changes sign near k_az ~ 0.45)
It is O(1): relative to the (divergent) St the error -> 0 only logarithmically, so at
physical slow-mode cutoffs it is a sizeable fraction of St. Whether it matters for the
observables (spectrum exponent f, flux) still needs propagation to those; f=1/2 is a
slow-mode (principal-value) quantity and is expected to be protected.

Created by Shahaf Aharony (Weizmann Institute) and Michal Shavit.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

W, F = 3.0, 0.5
SA = +1.0                      # alpha-leg chirality (compute St on the + branch)


def espec(kz, k, eps=0.0):
    # slow-mode regulator: |k_z|^{-F} -> (k_z^2 + eps^2)^{-F/2}
    return k ** (-W + F) * (kz * kz + eps * eps) ** (-F / 2)


def _omega_sum(x, z, rb, rg, sb, sg):
    """Omega(k_bz=x) = sa k_az k_a + sb k_bz k_b + sg k_gz k_g,  k_a=1."""
    kb = np.hypot(rb, x)
    kgz = -(z + x)
    kg = np.hypot(rg, kgz)
    return SA * z * 1.0 + sb * x * kb + sg * kgz * kg


def _omega_deriv(x, z, rb, rg, sb, sg):
    kb = np.hypot(rb, x)
    kg = np.hypot(rg, z + x)
    return sb * (rb * rb + 2 * x * x) / kb - sg * (rg * rg + 2 * (z + x) ** 2) / kg


def _roots(z, rb, rg, sb, sg, Xm, ngrid=400):
    xs = np.linspace(-Xm, Xm, ngrid)
    fs = _omega_sum(xs, z, rb, rg, sb, sg)
    out = []
    for i in range(ngrid - 1):
        if fs[i] == 0.0:
            out.append(xs[i])
        elif fs[i] * fs[i + 1] < 0:
            out.append(brentq(_omega_sum, xs[i], xs[i + 1], args=(z, rb, rg, sb, sg), xtol=1e-12))
    return out


def St(z, nt=48, nph=48, Tmax=7.0, eps=0.0):
    """Return (St_exact, St_approx) at k_alpha^z = z, |k_alpha|=1.

    eps > 0 applies a smooth slow-mode regulator to |k_z|^{-1/2} on all three legs.
    """
    rhoA2 = 1.0 - z * z
    rhoA = np.sqrt(rhoA2)
    ea = espec(z, 1.0, eps)
    xt, wt = leggauss(nt); t = 0.5 * Tmax * (xt + 1); wt = 0.5 * Tmax * wt
    xp, wp = leggauss(nph); ph = 0.5 * np.pi * xp; wp = 0.5 * np.pi * wp   # phi in (-pi/2,pi/2)
    Se = Sa = 0.0
    for i in range(nt):
        u = 0.5 * np.cosh(t[i]); sh = np.sinh(t[i])
        for j in range(nph):
            v = 0.5 * np.sin(ph[j]); co = np.cos(ph[j])
            rb = rhoA * (u + v); rg = rhoA * (u - v)
            A2D = (rhoA2 / 4.0) * co * sh                      # = (rho^2/4) sqrt((1-4v^2)(4u^2-1))
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
                        A3D = 0.25 * np.sqrt(a3s)
                        eb = espec(x, kb, eps); eg = espec(kgz, kg, eps)
                        Sspec = z * eb * eg + x * ea * eg + kgz * ea * eb
                        G2 = (A3D ** 2 / (4 * z * z * kb ** 2 * kg ** 2)
                              * (sb * kb - sg * kg) ** 2 * (SA * 1.0 + sb * kb + sg * kg) ** 2)
                        dOm = abs(_omega_deriv(x, z, rb, rg, sb, sg))
                        if dOm < 1e-14:
                            continue
                        common = z * G2 * Sspec / dOm          # k_az |Gamma|^2 S_spec / |dOmega|
                        # exact: 1/(2 A2D) * dudv-Jacobian(1/4 sh co) = 1/(2 rho^2)
                        Se += Wij * meas * common / (2 * rhoA2)
                        # approx: 1/(2 A3D) * (1/4 sh co)
                        Sa += Wij * meas * (0.25 * sh * co) * common / (2 * A3D)
    return Se, Sa


if __name__ == "__main__":
    print("Full St on the resonant manifold (w,f)=(3,1/2), sigma_alpha=+1")
    print(f"{'k_az':>7} {'St_exact':>13} {'St_approx':>13} {'rel.diff':>10}")
    for z in (0.1, 0.2, 0.3, 0.5, 0.7):
        Se, Sa = St(z)
        rd = (Se - Sa) / Se if Se != 0 else float("nan")
        print(f"{z:7.2f} {Se:13.5e} {Sa:13.5e} {rd:9.2%}")
