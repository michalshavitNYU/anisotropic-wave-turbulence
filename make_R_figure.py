"""Standalone SVG of R(k_z) for both spectra (odd-wave -> 1, isotropic -> pi/2)."""
import numpy as np
from ratio_quadrature import partial_N

kz = np.linspace(0.02, 0.97, 22)
Rodd = [partial_N(z, 0.5)[1] / partial_N(z, 0.5)[0] for z in kz]
Riso = [partial_N(z, 0.0)[1] / partial_N(z, 0.0)[0] for z in kz]

W, H, ml, mr, mt, mb = 480, 320, 60, 16, 16, 46
x0, x1, y0, y1 = ml, W - mr, H - mb, mt
xmin, xmax, ymin, ymax = 0.0, 1.0, 1.0, 3.0
sx = lambda x: x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
sy = lambda y: y0 + (y - ymin) / (ymax - ymin) * (y1 - y0)
def poly(ks, vs, col, dash=""):
    p = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in zip(ks, vs))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="2"{d}/>'
def tx(x, y, s, sz=12, an="middle", c="#222", rot=None):
    r = f' transform="rotate({rot} {x:.0f} {y:.0f})"' if rot is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{sz}" text-anchor="{an}" fill="{c}"{r}>{s}</text>'
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
s.append(f'<rect x="{x0}" y="{y1}" width="{x1-x0}" height="{y0-y1}" fill="none" stroke="#999"/>')
for yv in [1.0, 1.5, 2.0, 2.5, 3.0]:
    s.append(f'<line x1="{x0}" y1="{sy(yv):.1f}" x2="{x1}" y2="{sy(yv):.1f}" stroke="#eee"/>')
    s.append(tx(x0 - 6, sy(yv) + 4, f"{yv:.1f}", 11, "end"))
for xv in [0, 0.25, 0.5, 0.75, 1.0]:
    s.append(tx(sx(xv), y0 + 16, f"{xv:.2f}", 11))
s.append(f'<line x1="{x0}" y1="{sy(np.pi/2):.1f}" x2="{x1}" y2="{sy(np.pi/2):.1f}" stroke="#888780" stroke-dasharray="2,3"/>')
s.append(tx(sx(0.5), sy(np.pi/2) - 5, "pi/2", 10, "middle", "#888780"))
s.append(poly(kz, Riso, "#D85A30", "6,4"))
s.append(poly(kz, Rodd, "#378ADD"))
s.append(tx(sx(0.55), sy(Rodd[12]) + 16, "odd-wave  (-> 1)", 11, "start", "#378ADD"))
s.append(tx(sx(0.35), sy(Riso[8]) - 8, "isotropic  (-> pi/2)", 11, "start", "#D85A30"))
s.append(tx(W/2, H - 6, "k_z  (alpha leg;  k_z->0 = slow manifold)", 12))
s.append(tx(15, H/2, "R = <A3D> / <A2D>", 12, "middle", "#222", -90))
s.append("</svg>")
open("R_of_kz.svg", "w").write("\n".join(s))
print("odd-wave R(kz->0) ~", round(Rodd[0], 3), " isotropic ~", round(Riso[0], 3), "(pi/2 =", round(np.pi/2,3), ")")
print("wrote R_of_kz.svg")
