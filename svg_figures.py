"""Render the two figures as standalone SVG (no matplotlib needed)."""
import json, math

d = json.load(open("figure_data.json"))
EX = d["exact_odd"]


def frame(w=480, h=320, ml=62, mr=16, mt=16, mb=46):
    return dict(w=w, h=h, ml=ml, mr=mr, mt=mt, mb=mb,
                x0=ml, x1=w - mr, y0=h - mb, y1=mt)


def sx(f, x, xmin, xmax):
    return f["x0"] + (x - xmin) / (xmax - xmin) * (f["x1"] - f["x0"])


def sy(f, y, ymin, ymax):
    return f["y0"] + (y - ymin) / (ymax - ymin) * (f["y1"] - f["y0"])


def poly(pts, color, dash=""):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2"{da}/>'


def dots(pts, color):
    return "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/>' for x, y in pts)


def txt(x, y, s, size=12, anchor="middle", color="#222"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{color}">{s}</text>'


# ---------- Figure 1: ratio vs a ----------
f = frame()
xmin, xmax, ymin, ymax = 0.0, 0.5, 1.2, 2.6
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f["w"]} {f["h"]}">']
s.append(f'<rect x="{f["x0"]}" y="{f["y1"]}" width="{f["x1"]-f["x0"]:.0f}" height="{f["y0"]-f["y1"]:.0f}" fill="none" stroke="#999" stroke-width="1"/>')
for yv in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]:
    yy = sy(f, yv, ymin, ymax)
    s.append(f'<line x1="{f["x0"]}" y1="{yy:.1f}" x2="{f["x1"]}" y2="{yy:.1f}" stroke="#eee"/>')
    s.append(txt(f["x0"] - 6, yy + 4, f"{yv:.1f}", 11, "end"))
for xv in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    xx = sx(f, xv, xmin, xmax)
    s.append(txt(xx, f["y0"] + 16, f"{xv:.1f}", 11))
s.append(poly([(sx(f, x, xmin, xmax), sy(f, 2.0, ymin, ymax)) for x in [0, 0.5]], "#888780", "2,3"))
ptsA = [(sx(f, x, xmin, xmax), sy(f, y, ymin, ymax)) for x, y in zip(d["a_grid"], d["A_vals"])]
ptsB = [(sx(f, x, xmin, xmax), sy(f, y, ymin, ymax)) for x, y in zip(d["a_grid"], d["B_vals"])]
s.append(poly(ptsB, "#D85A30", "6,4")); s.append(dots(ptsB, "#D85A30"))
s.append(poly(ptsA, "#378ADD")); s.append(dots(ptsA, "#378ADD"))
s.append(txt(f["w"] / 2, f["h"] - 6, "anisotropy exponent  a   (e = k^-3+a |k_z|^-a)", 12))
s.append(txt(16, f["h"] / 2, "⟨A3D⟩ / ⟨A2D⟩", 12, "middle", "#222"))
s[-1] = f'<text x="16" y="{f["h"]/2:.0f}" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#222" transform="rotate(-90 16 {f["h"]/2:.0f})">⟨A3D⟩ / ⟨A2D⟩</text>'
s.append(txt(sx(f, 0.28, xmin, xmax), sy(f, 1.72, ymin, ymax) - 8, "ratio of averages", 11, "start", "#378ADD"))
s.append(txt(sx(f, 0.30, xmin, xmax), sy(f, 2.12, ymin, ymax) - 6, "average of ratio", 11, "start", "#D85A30"))
s.append("</svg>")
open("ratio_vs_a.svg", "w").write("\n".join(s))

# ---------- Figure 2: regulator convergence (log y) ----------
f = frame()
xmin, xmax = 0.0, 10.5
lymin, lymax = -5.0, -1.7   # log10 |err|
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {f["w"]} {f["h"]}">']
s.append(f'<rect x="{f["x0"]}" y="{f["y1"]}" width="{f["x1"]-f["x0"]:.0f}" height="{f["y0"]-f["y1"]:.0f}" fill="none" stroke="#999" stroke-width="1"/>')
for e in [-5, -4, -3, -2]:
    yy = sy(f, e, lymin, lymax)
    s.append(f'<line x1="{f["x0"]}" y1="{yy:.1f}" x2="{f["x1"]}" y2="{yy:.1f}" stroke="#eee"/>')
    s.append(txt(f["x0"] - 6, yy + 4, f"10^{e}", 11, "end"))
for xv in [0, 2, 4, 6, 8, 10]:
    xx = sx(f, xv, xmin, xmax)
    s.append(txt(xx, f["y0"] + 16, str(xv), 11))
def lerr(vals): return [max(abs(v - EX), 1e-5) for v in vals]
ph = [(sx(f, x, xmin, xmax), sy(f, math.log10(y), lymin, lymax)) for x, y in zip(d["h_grid"], lerr(d["h_vals"]))]
pd = [(sx(f, x, xmin, xmax), sy(f, math.log10(y), lymin, lymax)) for x, y in zip(d["d_grid"], lerr(d["d_vals"]))]
s.append(poly(pd, "#D85A30", "6,4")); s.append(dots(pd, "#D85A30"))
s.append(poly(ph, "#378ADD")); s.append(dots(ph, "#378ADD"))
s.append(txt(f["w"] / 2, f["h"] - 6, "regulator  (h or d)", 12))
s.append(f'<text x="16" y="{f["h"]/2:.0f}" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#222" transform="rotate(-90 16 {f["h"]/2:.0f})">| ratio(h,d) - 1.31104 |</text>')
s.append(txt(sx(f, 3.2, xmin, xmax), sy(f, math.log10(4e-4), lymin, lymax), "vary h (d=10)", 11, "start", "#378ADD"))
s.append(txt(sx(f, 5.5, xmin, xmax), sy(f, math.log10(3e-3), lymin, lymax), "vary d (h=6)", 11, "start", "#D85A30"))
s.append("</svg>")
open("regulator_convergence.svg", "w").write("\n".join(s))
print("wrote ratio_vs_a.svg, regulator_convergence.svg")
