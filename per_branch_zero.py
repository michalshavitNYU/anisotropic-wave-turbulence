"""Per-branch zeros of the very-anisotropic collision integral at the KZ point.
Result: BOTH channels vanish separately at (a,b)=(5/2,1/2):
  St_pos(2.5,0.5):  -1.2e-2 -> -1.2e-3 -> -1.2e-4   [(h,d)=(1,4),(2,6),(3,8)]
  St_rest(2.5,0.5): +1.8e-3 -> -7.9e-4 -> -1.1e-4
  root-found b*(a=2.5):  positive 0.4926,  rest 0.4992  (h=2,d=6) -> both 1/2.
Structural reason: leg permutations close the chirality triples into the two sets
{(+,+,+)} and {(+,+,-),(+,-,+),(+,-,-)}, so the KZ cancellation acts per set."""
from very_anisotropic_St import St
from scipy.optimize import brentq

POS = [(1, 1)]
REST = [(1, -1), (-1, 1), (-1, -1)]

if __name__ == "__main__":
    for h, d in [(1.0, 4), (2.0, 6), (3.0, 8)]:
        sp = St(2.5, 0.5, h=h, d=d, branches=POS)
        sr = St(2.5, 0.5, h=h, d=d, branches=REST)
        print(f"h={h} d={d}:  St_pos={sp:+.6e}   St_rest={sr:+.6e}", flush=True)
    for name, br in (("positive", POS), ("rest", REST)):
        bs = brentq(lambda b: St(2.5, b, h=2.0, d=6, branches=br), 0.3, 0.7, xtol=1e-6)
        print(f"b*(a=2.5), {name}: {bs:.6f}", flush=True)
