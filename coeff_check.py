"""Test the reduction St_alpha = |k_z|^{2b} C(b), b=-f.
C = St * |k_z|^{2f}.  Check (i) k_z-independence, (ii) C^(++) zero at f=1/2, C^rest not."""
import numpy as np
from collision_integral import St
POS = [(+1.0, +1.0)]
REST = [(-1.0, -1.0), (+1.0, -1.0), (-1.0, +1.0)]
eps = 1e-3; Tm = float(np.arccosh(2.0 / eps)) + 4.0; nt = 130
for label, br, nph in [("POS ", POS, 176), ("REST", REST, 128)]:
    for f in (0.40, 0.50, 0.60):
        row = []
        for z in (0.10, 0.20, 0.35):
            sa = St(z, nt=nt, nph=nph, Tmax=Tm, eps=eps, f=f, branches=br)[1]
            C = sa * z ** (2 * f)
            row.append((z, sa, C))
        s = "  ".join(f"z={z}: St={sa:+.3e} C={C:+.3e}" for z, sa, C in row)
        print(f"{label} f={f:.2f}:  {s}", flush=True)
