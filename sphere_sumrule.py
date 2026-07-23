"""Sum rule: for e = k^{-3} g(theta) with g=1 (w=3, f=0), the sphere-averaged
collision integral must vanish: int_0^1 St(z) dz = 0 (z = cos theta; St even in z,
phi trivial). Control: w=2.5 should NOT vanish."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from collision_integral import St

def stz(z, w, Tmax, nt=110, nph=80):
    return St(z, nt=nt, nph=nph, Tmax=Tmax, eps=0.0, f=0.0, w=w)[0]

print("1) leg-size convergence at f=0 (w=3):", flush=True)
for z in (0.05, 0.3, 0.8):
    for Tm in (6.0, 8.0, 10.0):
        print(f"   z={z} Tmax={Tm}: St={stz(z,3.0,Tm):+.5e}", flush=True)

print("2) small-z behavior (w=3):", flush=True)
for z in (0.02, 0.01, 0.005):
    print(f"   z={z}: St={stz(z,3.0,8.0):+.5e}", flush=True)

print("3) sphere average, GL-20 on z in (0,1):", flush=True)
x, wq = leggauss(20)
zs = 0.5 * (x + 1.0); ws = 0.5 * wq
for w in (3.0, 2.5):
    vals = np.array([stz(z, w, 8.0) for z in zs])
    I = float(np.sum(ws * vals))
    scale = float(np.sum(ws * np.abs(vals)))
    print(f"   w={w}:  int St dz = {I:+.5e}   int|St|dz = {scale:.5e}   ratio = {I/scale:+.3%}", flush=True)
