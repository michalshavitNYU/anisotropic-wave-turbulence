"""Approx (1/A3D) positive-branch St+ zero in a, eps=1e-5, several k_z."""
import numpy as np
from collision_integral import St
POS = [(+1.0, +1.0)]
eps = 1e-5; Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt, nph = 140, 144
a_grid = [0.50, 0.60, 0.70]
def cross(vals):
    for i in range(len(a_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            x0,x1,y0,y1 = a_grid[i],a_grid[i+1],vals[i],vals[i+1]
            return x0 - y0*(x1-x0)/(y1-y0)
    return None
print(f"approx St+ zero, eps={eps}, Tmax={Tm:.1f}, nph={nph}", flush=True)
for z in (0.20, 0.30, 0.40, 0.50):
    Ap = []
    for a in a_grid:
        sa = St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, branches=POS)[1]
        Ap.append(sa)
        print(f"  k_z={z} a={a:.2f}  approx={sa:+.4e}", flush=True)
    print(f"  --> k_z={z}: a* approx = {cross(Ap)}", flush=True)
