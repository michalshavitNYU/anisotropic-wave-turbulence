"""Sphere-averaged full St for e = k^{-3}|cos theta|^{-a}, smooth slow-mode
regulator eps. Find a*(eps) with I(a) = int_0^1 St(z) dz = 0 and track eps -> 0.
z-integral via z = t^2 (absorbs |z|^{-a} endpoint), GL-8 in t."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from collision_integral import St

x, wq = leggauss(8)
t = 0.5 * (x + 1.0); wt = 0.5 * wq

def I(a, eps, nph=72):
    Tm = float(np.arccosh(2.0 / eps)) + 4.0
    nt = int(7 * Tm)
    tot = 0.0
    for ti, wi in zip(t, wt):
        s = St(ti * ti, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=a, w=3.0)[0]
        tot += wi * 2.0 * ti * s
    return tot

a_grid = [0.40, 0.55, 0.70, 0.85]
for eps in (1e-2, 1e-3, 1e-4):
    vals = []
    for a in a_grid:
        v = I(a, eps); vals.append(v)
        print(f"eps={eps:.0e} a={a:.2f}: I={v:+.5e}", flush=True)
    found = False
    for i in range(len(a_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            a0, a1, y0, y1 = a_grid[i], a_grid[i + 1], vals[i], vals[i + 1]
            print(f"  --> a*(eps={eps:.0e}) = {a0 - y0 * (a1 - a0) / (y1 - y0):.4f}", flush=True)
            found = True
    if not found:
        print(f"  --> no crossing in [{a_grid[0]}, {a_grid[-1]}]", flush=True)
