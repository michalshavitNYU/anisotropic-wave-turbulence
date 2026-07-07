"""St+ zero in a for BOTH exact (1/A2D) and approx (1/A3D), positive branch, eps=1e-4."""
import numpy as np
from collision_integral import St

POS = [(+1.0, +1.0)]
eps = 1e-4
Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt, nph = 130, 160
a_grid = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

def cross(vals):
    for i in range(len(a_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            x0, x1, y0, y1 = a_grid[i], a_grid[i + 1], vals[i], vals[i + 1]
            return x0 - y0 * (x1 - x0) / (y1 - y0)
    return None

for z in (0.20, 0.40, 0.50):
    print(f"k_z = {z}:", flush=True)
    Ex, Ap = [], []
    for a in a_grid:
        se, sa = St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, branches=POS)
        Ex.append(se); Ap.append(sa)
        print(f"  a={a:.2f}  exact={se:+.4e}  approx={sa:+.4e}", flush=True)
    print(f"  a* exact  = {cross(Ex)}", flush=True)
    print(f"  a* approx = {cross(Ap)}", flush=True)
