"""Averaged-zero curve: w*(f) with int_0^1 St(z; w,f) dz = 0, full dispersion.
z-integral via z=t^2 (absorbs |z|^{-f}); GL-10 in t; eps=1e-3; Tmax spot-check."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from collision_integral import St

EPS = 1e-3
x, wq = leggauss(10)
t = 0.5 * (x + 1.0); wt = 0.5 * wq

def I(w, f, Tmax=8.0, nt=90, nph=80):
    tot = 0.0
    for ti, wi in zip(t, wt):
        z = ti * ti
        s = St(z, nt=nt, nph=nph, Tmax=Tmax, eps=EPS, f=f, w=w)[0]
        tot += wi * 2.0 * ti * s
    return tot

for f in (0.25, 0.5):
    print(f"f = {f}:", flush=True)
    vals = {}
    for w in (2.4, 2.7, 3.0):
        vals[w] = I(w, f)
        print(f"  w={w}: I={vals[w]:+.5e}", flush=True)
    ws = sorted(vals)
    for i in range(len(ws) - 1):
        a_, b_ = ws[i], ws[i + 1]
        if vals[a_] * vals[b_] < 0:
            wstar = a_ - vals[a_] * (b_ - a_) / (vals[b_] - vals[a_])
            print(f"  --> w*(f={f}) ~ {wstar:.3f}", flush=True)
    # Tmax spot-check at the middle w
    print(f"  spot-check w=2.7, Tmax=10: I={I(2.7, f, Tmax=10.0):+.5e}", flush=True)
