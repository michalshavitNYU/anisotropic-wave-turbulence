"""Rest branches with St_APPROX (index [1]); scout a > 0.5 for a zero."""
import numpy as np
from collision_integral import St
REST = [(-1.0, -1.0), (+1.0, -1.0), (-1.0, +1.0)]
eps = 1e-4; Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt, nph = 120, 96
a_grid = [0.50, 0.70, 0.90, 1.10, 1.30, 1.50]
print(f"REST St_approx scout (a>0.5), eps={eps}, nph={nph}", flush=True)
for z in (0.40, 0.50):
    print(f"k_z={z}:", flush=True)
    ap = []
    for a in a_grid:
        se, sa = St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, branches=REST)
        ap.append(sa)
        print(f"  a={a:.2f}  approx={sa:+.4e}  (exact={se:+.4e})", flush=True)
    for i in range(len(a_grid) - 1):
        if ap[i] * ap[i + 1] < 0:
            x0,x1,y0,y1 = a_grid[i],a_grid[i+1],ap[i],ap[i+1]
            print(f"  --> approx a* = {x0 - y0*(x1-x0)/(y1-y0):.3f}", flush=True)
