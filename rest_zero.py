"""Zero a* of the rest-of-branches sum (+,-,-),(+,+,-),(+,-,+) -- look at SMALL a."""
import numpy as np
from collision_integral import St
REST = [(-1.0, -1.0), (+1.0, -1.0), (-1.0, +1.0)]
eps = 1e-4; Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt, nph = 120, 96
a_grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
def cross(vals):
    for i in range(len(a_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            x0,x1,y0,y1 = a_grid[i],a_grid[i+1],vals[i],vals[i+1]
            return x0 - y0*(x1-x0)/(y1-y0)
    return None
print(f"REST zero (small a), eps={eps}, nph={nph}", flush=True)
for z in (0.40, 0.50):
    vals = []
    for a in a_grid:
        se = St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, branches=REST)[0]
        vals.append(se)
        print(f"  k_z={z} a={a:.2f}  St_rest={se:+.4e}", flush=True)
    print(f"  --> k_z={z}: a* = {cross(vals)}", flush=True)
