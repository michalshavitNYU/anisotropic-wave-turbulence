# Corrected appendix / response text — ⟨A₂D⟩ vs ⟨A₃D⟩

## 1. Corrected 3D triangle area

The area of the resonant triangle is `A₃D = ¼√[4 k_β² k_γ² − (k_β² + k_γ² − k_α²)²]`
with the **full** magnitudes `k_α² = ρ_α² + k_α^{z2}`. In the kinematic-box variables
the last term must therefore carry `−k_α^{z2}`:

```
A₃D = 1/4 · sqrt[ 4 ( ρ_α²(u+v)² + k_β^{z2} ) ( ρ_α²(u−v)² + (k_α^z+k_β^z)² )
                 − ( ρ_α²(2u²+2v²−1) − k_α^{z2} + k_β^{z2} + (k_α^z+k_β^z)² )² ]
```

(The previously circulated expression dropped the `−k_α^{z2}`, i.e. it used `ρ_α²` in
place of `k_α²`. This term is `O(k_z²)` — the same order as the `A₃D − A₂D` difference
being measured — so it is not negligible for this comparison.)

## 2. Quantity reported

For a spectrum-weighted measure `P ∝ e_p e_q δ(k+p+q)`, the geometric factor entering
`⟨Q₂D⟩ ≈ ⟨Q₃D⟩` is the **ratio of the averaged areas** (both `Q`'s are integrals):

```
⟨A₃D⟩/⟨A₂D⟩ = ∫ ⟨A₃D⟩_{βγ}(k_α^z) (k_α^z)^{-a} dk_α^z  /  ∫ ⟨A₂D⟩_{βγ}(k_α^z) (k_α^z)^{-a} dk_α^z
```

This is the well-conditioned quantity: it is finite, and in the isotropic limit it
equals exactly `2` (from `⟨|cos ψ|⟩ = 1/2` for a uniformly-oriented triangle normal),
independent of the radial spectrum. The alternative "average of the pointwise ratio"
`⟨A₃D/A₂D⟩` is dominated by an integrable singularity at `k_α^z → 1` (vertical leg,
`A₂D ∝ ρ_α² → 0`) and is not recommended.

## 3. Result for the family e = k^{-3+a}|k_z|^{-a}

Computed with a deterministic quadrature (exact singularity-removing substitutions;
converged to ~6 significant figures, and cross-checked against the regulated `(h,d)`
box, which converges to the same value):

| a    | spectrum             | ⟨A₃D⟩/⟨A₂D⟩ |
|------|----------------------|-------------|
| 0    | isotropic `k^{-3}`   | 2.0000      |
| 1/2  | odd-wave (this work) | 1.3110      |

The ratio decreases monotonically from the isotropic value `2` toward `1` as weight
concentrates on the slow-mode manifold, confirming that the approximation
`⟨A₂D⟩ ≈ ⟨A₃D⟩` is controlled precisely in the anisotropic regime of interest and
degrades (toward a factor of 2) as the distribution becomes isotropic — exactly the
behaviour anticipated by the referee. For the odd-wave spectrum the angularly-averaged
ratio is **≈ 1.31** (a ≈ 31% deviation), superseding the earlier `≈ 1.14` (which
combined the missing `−k_z²` with the average-of-ratio definition).

## 4. Regulator independence

The regulated box `u ∈ [½+10^{-d}, 10^h]`, etc., converges to the value above: the
outer cutoff `h` converges exponentially (essentially converged by `h ≈ 2`), the
branch-point cutoff `d` algebraically (`~10^{-d}`), consistent with the stability the
referee requested. See `regulator_convergence.svg`.

Figures: `ratio_vs_a.svg` (ratio vs a) and `regulator_convergence.svg`.
Code: `ratio_quadrature.py` (deterministic), `regulator_convergence.py` (regulated box).
