"""f* where full St_approx=0 vs hard slow-mode regulator heps, at k_z=0.2,0.6.
Test convergence f* -> 1/2 as heps -> 0."""
import numpy as np
from collision_integral import St
f_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
def fstar(z, heps):
    Tm = float(np.arccosh(2.0 / heps)) + 4.0; nt = int(8 * Tm)
    vals = [St(z, nt=nt, nph=110, Tmax=Tm, eps=0.0, heps=heps, f=f)[1] for f in f_grid]
    for i in range(len(f_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            x0, x1, y0, y1 = f_grid[i], f_grid[i + 1], vals[i], vals[i + 1]
            return x0 - y0 * (x1 - x0) / (y1 - y0), vals
    return None, vals
for z in (0.20, 0.60):
    print(f"k_z = {z}:", flush=True)
    for heps in (1e-2, 1e-3, 1e-4, 1e-5):
        fs, vals = fstar(z, heps)
        vv = " ".join(f"{v:+.2e}" for v in vals)
        print(f"  heps={heps:.0e}  f*={fs}   St(f): {vv}", flush=True)
