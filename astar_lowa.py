"""Low-a scan at FIXED Tmax=14.6 to isolate the eps-dependence of a*."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from collision_integral import St
x, wq = leggauss(8); t = 0.5*(x+1.0); wt = 0.5*wq
TM = 14.6; NT = int(7*TM)
def I(a, eps):
    return sum(wi*2.0*ti*St(ti*ti, nt=NT, nph=72, Tmax=TM, eps=eps, f=a, w=3.0)[0]
               for ti, wi in zip(t, wt))
a_grid = [0.10, 0.25, 0.40, 0.55]
for eps in (1e-2, 1e-3, 1e-4):
    vals = []
    for a in a_grid:
        v = I(a, eps); vals.append(v)
        print(f"eps={eps:.0e} a={a:.2f}: I={v:+.5e}", flush=True)
    for i in range(len(a_grid)-1):
        if vals[i]*vals[i+1] < 0:
            a0,a1,y0,y1 = a_grid[i],a_grid[i+1],vals[i],vals[i+1]
            print(f"  --> a*(eps={eps:.0e}) = {a0 - y0*(a1-a0)/(y1-y0):.4f}", flush=True)
