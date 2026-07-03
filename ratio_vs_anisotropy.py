"""
<A3D>/<A2D> for the family of energy spectra  e(k) = |k|^{-3+a} |k_z|^{-a}.

  a = 0    : isotropic  k^{-3}
  a = 1/2  : odd-wave spectrum  k^{-3}|cos theta|^{-1/2}  (the paper's case)

With k+p+q = 0 the triangle areas are exact cross products,
    A3D = 1/2 |k x p|         (= corrected Heron form, k_alpha^2 = 1)
    A2D = 1/2 |(k x p)_z|.
We also evaluate the "buggy" A3D that drops -k_alpha^{z2} (k_alpha^2 -> rho_alpha^2
= 1-k_z^2), to reproduce the manuscript's numbers.

At fixed |k_alpha| = 1 we importance-sample the leg p with a scale-invariant
proposal g(p) ~ |p|^{-3} (log-uniform radius x uniform direction), which keeps the
weights bounded in scale for any power-law spectrum. This gives the partial
integrals  N3(k_z) = int A3D e_p e_{|k+p|},  N2(k_z) = int A2D e_p e_{|k+p|}.

Two angularly-averaged ratios (weight (k_z)^{-a} = alpha-leg spectrum on the sphere):
  ratio-of-averages   A = int N3 (k_z)^{-a} dk_z / int N2 (k_z)^{-a} dk_z
  average-of-ratio    B = int (N3/N2)(k_z)^{-a} dk_z / int (k_z)^{-a} dk_z
The angular integral is done in theta (k_z = cos theta), which regularizes the
integrable R -> inf singularity at k_z -> 1 (vertical k_alpha, A2D ~ rho_alpha^2 -> 0).

Created by Shahaf Aharony (Weizmann Institute) and Michal Shavit.
"""
import numpy as np


def espec(v, a):
    r = np.linalg.norm(v, axis=1)
    vz = np.abs(v[:, 2])
    return r ** (-3.0 + a) * vz ** (-a)


def partial_integrals(kz, a, N, rng, rmin=1e-4, rmax=1e4, nbatch=6):
    """N3_correct, N3_buggy, N2 at fixed |k_alpha|=1, angle with k_z=cos theta."""
    k = np.array([np.sqrt(max(0.0, 1 - kz ** 2)), 0.0, kz])
    L = np.log(rmax / rmin)
    n3c = n3b = n2 = 0.0
    for _ in range(nbatch):
        r = np.exp(rng.uniform(np.log(rmin), np.log(rmax), N))
        d = rng.normal(size=(N, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = r[:, None] * d
        w = espec(p, a) * espec(k + p, a) * (4 * np.pi * L * r ** 3)  # e_p e_{k+p}/g
        cx = np.cross(k, p)
        n2 += np.sum(0.5 * np.abs(cx[:, 2]) * w)
        n3c += np.sum(0.5 * np.linalg.norm(cx, axis=1) * w)
        kb2 = np.sum(p ** 2, axis=1)
        kg2 = np.sum((k + p) ** 2, axis=1)
        a3b = 4 * kb2 * kg2 - (kb2 + kg2 - (1 - kz ** 2)) ** 2   # buggy: k_alpha^2 -> rho^2
        n3b += np.sum(0.25 * np.sqrt(np.clip(a3b, 0, None)) * w)
    return n3c, n3b, n2


def averaged_ratios(a, N=1_500_000, ntheta=80, seed=1):
    rng = np.random.default_rng(seed)
    theta = (np.arange(ntheta) + 0.5) / ntheta * (np.pi / 2)   # (0, pi/2)
    kz = np.cos(theta)
    wgt = kz ** (-a) * np.sin(theta)                            # (k_z)^{-a} dk_z  in theta
    N3c = np.zeros(ntheta); N3b = np.zeros(ntheta); N2 = np.zeros(ntheta)
    for i, z in enumerate(kz):
        N3c[i], N3b[i], N2[i] = partial_integrals(z, a, N, rng)
    Rc, Rb = N3c / N2, N3b / N2
    out = dict(
        A_corr=np.sum(N3c * wgt) / np.sum(N2 * wgt),
        A_buggy=np.sum(N3b * wgt) / np.sum(N2 * wgt),
        B_corr=np.sum(Rc * wgt) / np.sum(wgt),
        B_buggy=np.sum(Rb * wgt) / np.sum(wgt),
        kz=kz, Rc=Rc, Rb=Rb,
    )
    return out


if __name__ == "__main__":
    for a in (0.5, 0.0):
        r = averaged_ratios(a)
        print(f"\n=== a = {a}   e = k^(-3+a) |k_z|^(-a) ===")
        print(f"  ratio-of-averages  <A3D>/<A2D> :  corrected = {r['A_corr']:.3f}   buggy = {r['A_buggy']:.3f}")
        print(f"  average-of-ratio   <A3D/A2D>_kz:  corrected = {r['B_corr']:.3f}   buggy = {r['B_buggy']:.3f}")
