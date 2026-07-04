"""
High-accuracy deterministic quadrature for the partial averages

    N2(k_z) = <A2D>_{beta gamma},   N3(k_z) = <A3D>_{beta gamma}

of the family e(k) = |k|^{-3+a} |k_z|^{-a}, at fixed |k_alpha| = 1 (k_z = cos theta).
Corrected A3D (Heron with k_alpha^2 = 1).

Reduction (paper's (u,v,k_beta^z) box):
  A2D = (rho_a^2/4) sqrt((1-4v^2)(4u^2-1))   -> exactly factorized.
Substitutions that make the integrand smooth:
  v = 1/2 sin(phi),  phi in (-pi/2,pi/2)      (removes sqrt(1-4v^2))
  u = 1/2 cosh(t),   t in (0, inf)            (removes sqrt(4u^2-1); in N3 the 1/A2D
                                               cancels the Jacobian outright)
  k_beta^z branch points |k_b^z|^{-a}, |k_a^z+k_b^z|^{-a} at k_b^z = 0 and -k_a^z:
     middle (-k_z,0): k_b^z = -k_z/2 + (k_z/2) sin(xi)   (removes BOTH, for a<=1/2)
     right  (0, inf): k_b^z = s^2,      s = r/(1-r)       (removes sqrt at 0 + tail)
     left   (-inf,-k_z): k_b^z = -k_z - s^2, s = r/(1-r)
Densities in the substituted variables:
  N2 = rho_a^4 * 1/4 (u^2-v^2) sinh(t) cos(phi) * W
  N3 = rho_a^2 * (u^2-v^2) * A3D * W
with W = |k_b^z|^{-a}|k_g^z|^{-a} (rho_a^2(u+v)^2+k_b^{z2})^{-(3-a)/2}
                                 (rho_a^2(u-v)^2+k_g^{z2})^{-(3-a)/2},  k_g^z = k_a^z+k_b^z.

Tensor Gauss-Legendre (t truncated at Tmax, exponential decay). Accuracy is checked
by node-count doubling; a relative error is returned with every number.

Created by Shahaf Aharony (Weizmann Institute) and Michal Shavit.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def _gl(n, a_, b_):
    x, w = leggauss(n)
    return 0.5 * (b_ - a_) * x + 0.5 * (b_ + a_), 0.5 * (b_ - a_) * w


def partial_N(z, a, nt=64, nph=56, nkb=64, Tmax=26.0):
    """Return N2(z), N3(z) via tensor GL quadrature."""
    rhoA2 = 1.0 - z * z

    t, wt = _gl(nt, 0.0, Tmax)
    ph, wph = _gl(nph, -np.pi / 2, np.pi / 2)
    u = 0.5 * np.cosh(t); v = 0.5 * np.sin(ph)
    u2 = u * u; v2 = v * v
    U2V2 = u2[:, None] - v2[None, :]          # (nt,nph)
    UPV = (u[:, None] + v[None, :])
    UMV = (u[:, None] - v[None, :])
    SH = np.sinh(t)[:, None] + 0.0 * ph[None, :]
    CO = np.cos(ph)[None, :] + 0.0 * u[:, None]
    P2 = (2 * u2[:, None] + 2 * v2[None, :] - 1.0)
    wtph = wt[:, None] * wph[None, :]          # (nt,nph)

    N2 = 0.0; N3 = 0.0
    for seg in ("M", "R", "L"):
        if seg == "M":
            # k_b^z = -z/2 + z/2 sin(xi): the singular |k_b^z|^{-a}|k_g^z|^{-a} times the
            # Jacobian is (z/2)^{1-2a} cos(xi)^{1-2a} in closed form (=1 for a=1/2).
            xi, wk = _gl(nkb, -np.pi / 2, np.pi / 2)
            c = np.cos(xi)
            kb = -z / 2 + (z / 2) * np.sin(xi)
            SJ = wk * (z / 2) ** (1 - 2 * a) * c ** (1 - 2 * a)   # branch x Jacobian x weight
            brf = 1.0                                            # remaining finite branch factor
        else:
            r, wr = _gl(nkb, 0.0, 1.0)
            s = r / (1.0 - r); dsdr = 1.0 / (1.0 - r) ** 2
            SJ = wr * 2.0 * s ** (1 - 2 * a) * dsdr               # |branch|^{-a} x Jacobian x weight
            if seg == "R":
                kb = s * s                                       # k_b^z -> 0 branch handled by SJ
                brf = np.abs(z + kb) ** (-a)                      # finite |k_g^z|^{-a}
            else:
                kb = -z - s * s                                  # k_g^z -> 0 branch handled by SJ
                brf = np.abs(z + s * s) ** (-a)                  # finite |k_b^z|^{-a}
        kb = kb[None, None, :]; kg = z + kb
        upv = UPV[:, :, None]; umv = UMV[:, :, None]
        beta2 = rhoA2 * upv ** 2 + kb ** 2
        gam2 = rhoA2 * umv ** 2 + kg ** 2
        W = beta2 ** (-(3 - a) / 2) * gam2 ** (-(3 - a) / 2) * brf
        X = rhoA2 * P2[:, :, None] - z * z + kb ** 2 + kg ** 2
        a3s = 4.0 * beta2 * gam2 - X ** 2
        A3D = 0.25 * np.sqrt(np.clip(a3s, 0.0, None))

        base = wtph[:, :, None] * SJ[None, None, :] * W
        N2 += rhoA2 ** 2 * 0.25 * np.sum(base * U2V2[:, :, None] * SH[:, :, None] * CO[:, :, None])
        N3 += rhoA2 * np.sum(base * U2V2[:, :, None] * A3D)
    return N2, N3


def N_converged(z, a, tol=1e-6):
    """N2, N3 with a relative-error estimate from node doubling."""
    lo = partial_N(z, a, nt=48, nph=44, nkb=48)
    hi = partial_N(z, a, nt=72, nph=64, nkb=72)
    err2 = abs(hi[0] - lo[0]) / abs(hi[0])
    err3 = abs(hi[1] - lo[1]) / abs(hi[1])
    return hi[0], hi[1], max(err2, err3)


def averaged_ratios(a, nt=80, nph=64, nkb=80):
    """Angle-averaged ratios over k_z in (0,1), weight (k_z)^{-a} = e_alpha.

    A (ratio-of-averages) = int N3 z^{-a} dz / int N2 z^{-a} dz
    B (average-of-ratio)  = int (N3/N2) z^{-a} dz / int z^{-a} dz,   int z^{-a}dz = 1/(1-a)
    Singularities handled analytically: z^{-a} at z=0 and (1-z)^{-1/2} in R at z=1.
    """
    from scipy.integrate import quad

    def _clip(z):
        return min(max(z, 1e-12), 1.0 - 1e-12)

    def N2(z):
        return partial_N(_clip(z), a, nt, nph, nkb)[0]

    def N3(z):
        return partial_N(_clip(z), a, nt, nph, nkb)[1]

    def Rsmooth(z):  # R(z)*sqrt(1-z), smooth (removes the (1-z)^{-1/2} in R)
        n2, n3 = partial_N(_clip(z), a, nt, nph, nkb)
        return (n3 / n2) * np.sqrt(1.0 - z)

    qk = dict(epsabs=1e-10, epsrel=1e-9, limit=100)
    num_A, eA = quad(N3, 0, 1, weight="alg", wvar=(-a, 0.0), **qk)
    den_A, eA2 = quad(N2, 0, 1, weight="alg", wvar=(-a, 0.0), **qk)
    num_B, eB = quad(Rsmooth, 0, 1, weight="alg", wvar=(-a, -0.5), **qk)
    den_B = 1.0 / (1.0 - a)
    A = num_A / den_A
    B = num_B / den_B
    A_err = A * (abs(eA / num_A) + abs(eA2 / den_A))
    B_err = abs(eB / den_B)
    return A, A_err, B, B_err


if __name__ == "__main__":
    print("Validation of R(k_z) = N3/N2  (a = 1/2), vs Monte Carlo:")
    print(f"{'k_z':>6} {'N2':>12} {'N3':>12} {'R':>9} {'rel.err':>9}   MC R")
    mc = {0.043: 1.185, 0.162: 1.281, 0.357: 1.453, 0.628: 1.817}
    for z in (0.043, 0.162, 0.357, 0.628):
        N2v, N3v, e = N_converged(z, 0.5)
        print(f"{z:6.3f} {N2v:12.5e} {N3v:12.5e} {N3v / N2v:9.4f} {e:9.1e}   {mc[z]:.3f}")

    print("\nAngle-averaged <A3D>/<A2D>  (deterministic quadrature, high accuracy):")
    for a in (0.5, 0.0):
        A, Ae, B, Be = averaged_ratios(a)
        print(f"  a = {a}:  ratio-of-averages = {A:.5f} (+/- {Ae:.1e})"
              f"    average-of-ratio = {B:.5f} (+/- {Be:.1e})")
