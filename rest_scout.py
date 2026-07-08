"""Sum of the non-positive branches: (s_b,s_g) in {(+,-),(-,+),(-,-)}. Locate zero in a."""
import numpy as np
from collision_integral import St
REST = [(+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0)]
eps = 1e-4; Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt, nph = 120, 96
a_grid = [0.30, 0.50, 0.70, 0.90, 1.10]
print(f"REST branches, eps={eps}, Tmax={Tm:.1f}, nph={nph}", flush=True)
for z in (0.40, 0.50):
    print(f"k_z={z}:", flush=True)
    for a in a_grid:
        se = St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, branches=REST)[0]
        print(f"  a={a:.2f}  St_rest={se:+.4e}", flush=True)
