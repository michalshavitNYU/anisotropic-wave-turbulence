"""Zero of positive-branch St^+_exact in a, at several fixed k_z (eps=1e-4).
Maps the St+=0 contour in the (k_z, a) plane."""
import numpy as np
from collision_integral import St

POS = [(+1.0, +1.0)]
eps = 1e-4
Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt, nph = 130, 160
a_grid = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

def stp(z, a):
    return St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, branches=POS)[0]

for z in (0.20, 0.40, 0.50):
    print(f"k_z = {z}:", flush=True)
    vals = []
    for a in a_grid:
        v = stp(z, a); vals.append(v)
        print(f"  a={a:.2f}  St+={v:+.4e}", flush=True)
    crossed = False
    for i in range(len(a_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            x0, x1, y0, y1 = a_grid[i], a_grid[i + 1], vals[i], vals[i + 1]
            print(f"  --> a* = {x0 - y0 * (x1 - x0) / (y1 - y0):.3f}", flush=True)
            crossed = True
    if not crossed:
        print("  (no sign change in [0.2,0.8])", flush=True)
