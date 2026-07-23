"""Pointwise full St(z) at fixed eps=1e-3, scanning high angular exponents a."""
import numpy as np
from collision_integral import St
eps = 1e-3
Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt = int(7 * Tm)
a_grid = [0.85, 1.0, 1.2, 1.4, 1.6]
for z in (0.2, 0.4, 0.6, 0.8):
    vals = []
    for a in a_grid:
        s = St(z, nt=nt, nph=80, Tmax=Tm, eps=eps, f=a, w=3.0)[0]
        vals.append(s)
        print(f"z={z} a={a:.2f}: St={s:+.5e}", flush=True)
    for i in range(len(a_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            a0, a1, y0, y1 = a_grid[i], a_grid[i + 1], vals[i], vals[i + 1]
            print(f"  --> zero at z={z}: a* = {a0 - y0*(a1-a0)/(y1-y0):.3f}", flush=True)
