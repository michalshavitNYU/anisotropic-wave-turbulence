"""Compute data for the ratio-vs-a and regulator-convergence figures (no matplotlib)."""
import json
from ratio_quadrature import averaged_ratios
from regulator_convergence import ratio_of_averages_reg

EXACT_ODD = 1.311037

a_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
A_vals, B_vals = [], []
for a in a_grid:
    A, _, B, _ = averaged_ratios(a, nt=64, nph=52, nkb=64)
    A_vals.append(A); B_vals.append(B)
    print(f"a={a:.2f}  A(ratio-of-avg)={A:.5f}  B(avg-of-ratio)={B:.5f}", flush=True)

h_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
h_vals = [ratio_of_averages_reg(0.5, h, 10) for h in h_grid]
print("h-sweep(d=10):", [round(v,5) for v in h_vals], flush=True)
d_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10]
d_vals = [ratio_of_averages_reg(0.5, 6.0, d) for d in d_grid]
print("d-sweep(h=6): ", [round(v,5) for v in d_vals], flush=True)

json.dump(dict(a_grid=a_grid, A_vals=A_vals, B_vals=B_vals, exact_odd=EXACT_ODD,
               h_grid=h_grid, h_vals=h_vals, d_grid=d_grid, d_vals=d_vals),
          open("figure_data.json", "w"), indent=1)
print("wrote figure_data.json")
