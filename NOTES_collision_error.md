# Collision-integral error of the ⟨A2D⟩≈⟨A3D⟩ approximation

## R(k_z) is ~1 where the physics lives  (slow_mode_and_I32_check.py, task 1)

R(k_z) = ⟨A3D⟩_βγ / ⟨A2D⟩_βγ at fixed α-leg angle (a=1/2):

  k_z:  0.001  0.05  0.2   0.3   0.5   0.7   0.9   0.97
  R  :  1.09   1.19  1.32  1.40  1.64  2.08  3.54  6.43

R → 1 as k_z → 0 (slow-mode manifold), where the spectrum |cosθ|^{-1/2} concentrates
the energy and flux. So the approximation is exact where it is used; the flat angular
average (1.31) is only reached near k_z ≈ 0.2 and OVERSTATES the physical error.

## I32 alone is NOT the collision error — it diverges (task 2)

I32 = ⟨A3D^2/A2D⟩_βγ (the June-21 quantity) diverges logarithmically at the collinear
edge v → ±1/2, where the horizontal triangle degenerates (ρ_β = ρ_α+ρ_γ, A2D→0) but the
3D triangle stays finite (A3D↛0):

  I32(k_z=0.3) vs edge cutoff d:  25.6 (d2), 46.4 (d4), 64.3 (d6), 70.8 (d7)  -> grows ~ d

So I32 is regulator-dependent and cannot be quoted as the collision error by itself.

## The correct quantity

St^exact / St^approx  (swap 1/A2D -> 1/A3D in the collision integral) is well-defined only
with the FULL resonant integrand: the frequency delta δ(Σ s_i k_i^z k_i) FIXES k_β^z on the
resonant manifold (not integrated freely as in the geometric average), and the vertex
prefactor (s_β k_β - s_γ k_γ)^2 (s_α k_α + s_β k_β + s_γ k_γ)^2 suppresses the collinear
edge. Recommended route: run the existing Julia flux/collision code twice (1/A2D vs 1/A3D)
and compare St(k_z) or the flux Π — the validated machinery already carries δ and the
prefactor.
