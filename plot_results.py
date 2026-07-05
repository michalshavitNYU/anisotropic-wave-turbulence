"""
Figures for the response letter:
  (1) ratio_vs_a.pdf         : <A3D>/<A2D> vs anisotropy a, e=k^(-3+a)|k_z|^(-a).
  (2) regulator_convergence.pdf : regulated (h,d) box integral -> exact value.
Also prints the underlying data.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ratio_quadrature import averaged_ratios
from regulator_convergence import ratio_of_averages_reg

EXACT_ODD = 1.311037  # ratio-of-averages at a=1/2 (substitution method, converged)

# ---------- (1) ratio vs a ----------
a_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
A_vals, B_vals = [], []
for a in a_grid:
    A, Ae, B, Be = averaged_ratios(a, nt=64, nph=52, nkb=64)
    A_vals.append(A); B_vals.append(B)
    print(f"a={a:.2f}  ratio-of-averages={A:.5f}  average-of-ratio={B:.5f}")

# ---------- (2) regulator convergence (a=1/2) ----------
h_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
h_vals = [ratio_of_averages_reg(0.5, h, 10) for h in h_grid]   # d=10 fixed
d_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10]
d_vals = [ratio_of_averages_reg(0.5, 6.0, d) for d in d_grid]   # h=6 fixed
print("\nh-sweep (d=10):", [f"{v:.5f}" for v in h_vals])
print("d-sweep (h=6): ", [f"{v:.5f}" for v in d_vals])

data = dict(a_grid=a_grid, A_vals=A_vals, B_vals=B_vals, exact_odd=EXACT_ODD,
            h_grid=h_grid, h_vals=h_vals, d_grid=d_grid, d_vals=d_vals)
with open("figure_data.json", "w") as f:
    json.dump(data, f, indent=1)

# ---------- plot 1 ----------
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot(a_grid, A_vals, "o-", label=r"ratio of averages $\langle A_{3D}\rangle/\langle A_{2D}\rangle$")
ax.plot(a_grid, B_vals, "s--", label=r"average of ratio $\langle A_{3D}/A_{2D}\rangle$")
ax.axhline(2.0, color="gray", ls=":", lw=1)
ax.text(0.02, 2.02, "isotropic limit = 2", fontsize=8, color="gray")
ax.set_xlabel(r"anisotropy exponent $a$  ($e=k^{-3+a}|k_z|^{-a}$)")
ax.set_ylabel(r"$\langle A_{3D}\rangle/\langle A_{2D}\rangle$")
ax.set_title("Area ratio vs. anisotropy")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("ratio_vs_a.pdf"); fig.savefig("ratio_vs_a.png", dpi=140)

# ---------- plot 2 ----------
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot(h_grid, np.abs(np.array(h_vals) - EXACT_ODD), "o-", label="vary $h$ (regulator, $d=10$)")
ax.plot(d_grid, np.abs(np.array(d_vals) - EXACT_ODD), "s--", label="vary $d$ (regulator, $h=6$)")
ax.set_yscale("log")
ax.set_xlabel(r"regulator ($h$ or $d$)")
ax.set_ylabel(r"$|\,$ratio$(h,d) - 1.31104\,|$")
ax.set_title("Convergence of regulated box to exact value")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("regulator_convergence.pdf"); fig.savefig("regulator_convergence.png", dpi=140)
print("\nsaved: ratio_vs_a.pdf/png, regulator_convergence.pdf/png, figure_data.json")
