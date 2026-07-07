"""Fix k_z=0.3, eps=1e-4; sweep angular exponent a; find where St^+_exact = 0."""
import numpy as np
from collision_integral import St

POS = [(+1.0, +1.0)]
z, eps = 0.3, 1e-4
Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt, nph = 135, 176

def stp(a):
    return St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, branches=POS)[0]

print(f"St^+_exact(a) at k_z={z}, eps={eps}, nph={nph}, Tmax={Tm:.1f}:", flush=True)
a_grid = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
vals = []
for a in a_grid:
    v = stp(a); vals.append(v)
    print(f"  a={a:.2f}  St+={v:+.4e}", flush=True)
for i in range(len(a_grid) - 1):
    if vals[i] * vals[i + 1] < 0:
        x0, x1, y0, y1 = a_grid[i], a_grid[i + 1], vals[i], vals[i + 1]
        astar = x0 - y0 * (x1 - x0) / (y1 - y0)
        print(f"  --> zero crossing near a* = {astar:.3f}", flush=True)
