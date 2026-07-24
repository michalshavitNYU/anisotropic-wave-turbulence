"""Full-dispersion flux pieces at w=3, e = k^-3 |cos theta|^-a, eps fixed.
P(z,a) = (1/2) dSt/dw |_{w=3}  (finite/'constant-flux' part, central diff dw=0.05)
D(z,a) = -St(z; w=3)           (drain = coefficient of ln k; 0 iff scale-invariant)"""
import numpy as np
from collision_integral import St
EPS = 1e-3
TM = float(np.arccosh(2.0 / EPS)) + 4.0
NT, NPH, DW = int(7 * TM), 80, 0.05

def pieces(z, a):
    sp = St(z, nt=NT, nph=NPH, Tmax=TM, eps=EPS, f=a, w=3.0 + DW)[0]
    s0 = St(z, nt=NT, nph=NPH, Tmax=TM, eps=EPS, f=a, w=3.0)[0]
    sm = St(z, nt=NT, nph=NPH, Tmax=TM, eps=EPS, f=a, w=3.0 - DW)[0]
    return 0.5 * (sp - sm) / (2 * DW), -s0

for z in (0.3, 0.6):
    print(f"z = {z}:", flush=True)
    print(f"  {'a':>5} {'P=(1/2)dSt/dw':>15} {'D=-St(3)':>12}", flush=True)
    for a in (0.30, 0.40, 0.50, 0.60, 0.70):
        P, D = pieces(z, a)
        print(f"  {a:5.2f} {P:15.5f} {D:12.5f}", flush=True)
