"""High-a pointwise St at eps=1e-4 (dip region focus)."""
import numpy as np
from collision_integral import St
eps = 1e-4
Tm = float(np.arccosh(2.0 / eps)) + 4.0
nt = int(7 * Tm)
for z in (0.4, 0.6, 0.8):
    prev = None
    for a in (1.2, 1.4, 1.6, 1.8, 2.0):
        s = St(z, nt=nt, nph=80, Tmax=Tm, eps=eps, f=a, w=3.0)[0]
        print(f"z={z} a={a:.2f}: St={s:+.5e}", flush=True)
        if prev is not None and prev * s < 0:
            print(f"  --> SIGN CHANGE at z={z} between a={a-0.2:.1f} and {a:.1f}", flush=True)
        prev = s
