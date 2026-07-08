"""Track the rest-branches minimum (near a~0.2) as eps->0: does a zero appear?"""
import numpy as np
from collision_integral import St
REST = [(-1.0, -1.0), (+1.0, -1.0), (-1.0, +1.0)]
z = 0.40
for eps in (1e-4, 1e-5, 1e-6):
    Tm = float(np.arccosh(2.0 / eps)) + 4.0
    nt = int(8 * Tm)
    print(f"eps={eps:.0e}, Tmax={Tm:.1f}:", flush=True)
    for a in (0.10, 0.20, 0.30):
        se = St(z, nt=nt, nph=96, Tmax=Tm, eps=eps, f=a, branches=REST)[0]
        print(f"  a={a:.2f}  St_rest={se:+.4e}", flush=True)
