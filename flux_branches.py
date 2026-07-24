"""Per-branch radial energy flux at the KZ point, very anisotropic limit.
Pi^br ∝ (1/2) dSt^br/dw |_(w=3, f=1/2) = (1/2) dSt^br/da at (a,b)=(5/2,1/2)
(a = w - f).  Valid per branch because each channel's St vanishes separately
at the KZ point. Paper's claim: Pi(+,+) < 0 (same-helicity backscatter),
Pi(+,-) = Pi(-,+) > 0 (forward), Pi(-,-) smallest; total > 0 (direct cascade)."""
import numpy as np
from very_anisotropic_St import St

BR = {"(+,+)": [(1, 1)], "(+,-)": [(1, -1)], "(-,+)": [(-1, 1)], "(-,-)": [(-1, -1)]}

def flux(branches, h, d, da=0.05):
    sp = St(2.5 + da, 0.5, h=h, d=d, branches=branches)
    sm = St(2.5 - da, 0.5, h=h, d=d, branches=branches)
    return 0.5 * (sp - sm) / (2 * da)

for h, d in [(2.0, 6), (3.0, 8)]:
    print(f"(h,d)=({h},{d}):", flush=True)
    tot = 0.0
    for name, br in BR.items():
        P = flux(br, h, d)
        tot += P
        print(f"  Pi{name} = {P:+.5f}", flush=True)
    print(f"  Pi_total = {tot:+.5f}", flush=True)
# step-size check
print("step-size check (h=2,d=6), da=0.1:", flush=True)
for name, br in BR.items():
    print(f"  Pi{name} = {flux(br, 2.0, 6, da=0.1):+.5f}", flush=True)
