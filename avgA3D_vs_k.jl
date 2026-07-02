# Partial average of the 3D triangle area over the two "leg" wavevectors.
#
# Computes  <A3D>_{beta,gamma}(k_alpha)  =  int dV_beta dV_gamma  A3D  e_beta e_gamma
#                                                  * delta(k_alpha + k_beta + k_gamma),
# i.e. A3D averaged over p = k_beta and q = k_gamma against the distribution
# e_p e_q delta(p + k + q). By homogeneity this is a function of the alpha leg only;
# we fix |k_alpha| = 1 and evaluate it as a function of k_alpha^z = cos(theta).
#
# Areas (Heron), with the corrected A3D (note the -kaz^2 term, which the earlier
# code and the manuscript appendix were missing):
#   A2D = (rho_alpha^2 / 4) sqrt[ 4(u+v)^2 (u-v)^2 - (2u^2 + 2v^2 - 1)^2 ]
#   A3D = (1/4) sqrt[ 4 k_beta^2 k_gamma^2 - (k_beta^2 + k_gamma^2 - k_alpha^2)^2 ],
#         k_alpha^2 = rho_alpha^2 + kaz^2  (so k_beta^2 + k_gamma^2 - k_alpha^2 carries -kaz^2).
#
# Kinematic-box variables (rho_beta = rho_alpha (u+v), rho_gamma = rho_alpha (u-v),
# k_gamma^z = -(kaz + kbz)); the azimuthal delta functions contribute I_phi = 1/(2 A2D),
# and the (rho_beta, rho_gamma) -> (u,v) Jacobian gives 2 rho_alpha^4 (u+v)(u-v), so that
#   <A3D>_{beta gamma} = rho_alpha^4 int du dv dkbz (u+v)(u-v) (A3D/A2D) e_beta e_gamma.
#
# Created by Shahaf Aharony (Weizmann Institute) and Michal Shavit.
# Area correction and swappable-spectrum generalization added on top of area2Dvs3_v3.jl.

using HCubature
# using Plots, LaTeXStrings, DataFrames, CSV   # optional: plotting / saving

# ------------------------------------------------------------------------------------
# Spectrum e(k), written as a function of (rho^2, kz). Swap this single function to
# compare different probability distributions P ~ e_p e_q delta(p+k+q).
#   Odd-wave spectrum:  e = |kz|^{-1/2} (rho^2 + kz^2)^{-5/4}  ==  k^{-3} |cos theta|^{-1/2}.
# ------------------------------------------------------------------------------------
espec_odd(rho2, kz) = abs(kz)^(-1 / 2) * (rho2 + kz^2)^(-5 / 4)

# A couple of alternatives, ready for the distribution comparison (not used by default):
espec_iso(rho2, kz) = 1.0                                   # e = 1 (thermal / isotropic)
espec_pow(rho2, kz; s = 5 / 4) = (rho2 + kz^2)^(-s)         # isotropic power law
espec_gen(rho2, kz; s = 5 / 4, a = 1 / 2) = abs(kz)^(-a) * (rho2 + kz^2)^(-s)  # tunable

# ------------------------------------------------------------------------------------
# Areas in the (u, v, kbz) kinematic box, at fixed alpha leg with |k_alpha| = 1.
# ------------------------------------------------------------------------------------
a2Dsqrt(rhoA2, u, v) = 4 * (u + v)^2 * (u - v)^2 - (2 * u^2 + 2 * v^2 - 1)^2

function a3Dsqrt(rhoA2, kaz, kbz, u, v)
    4 * (rhoA2 * (u + v)^2 + kbz^2) * (rhoA2 * (u - v)^2 + (kaz + kbz)^2) -
    (rhoA2 * (2 * u^2 + 2 * v^2 - 1) - kaz^2 + kbz^2 + (kaz + kbz)^2)^2
end

# Integrand of <A3D>_{beta gamma} in (u, v, kbz); spectrum e is swappable.
function integrandA3D(kaz, kbz, u, v; e = espec_odd)
    rhoA2 = 1 - kaz^2                       # rho_alpha^2, since |k_alpha| = 1
    a2s = a2Dsqrt(rhoA2, u, v)
    a3s = a3Dsqrt(rhoA2, kaz, kbz, u, v)
    (a2s > 0 && a3s > 0) || return 0.0
    A2D = rhoA2 * sqrt(a2s) / 4
    A3D = sqrt(a3s) / 4
    eb = e(rhoA2 * (u + v)^2, kbz)          # e_beta  (rho_beta^2 = rhoA2 (u+v)^2)
    eg = e(rhoA2 * (u - v)^2, kaz + kbz)    # e_gamma (k_gamma^z = -(kaz+kbz))
    return rhoA2^2 * (u + v) * (u - v) * (A3D / A2D) * eb * eg
end

# Same measure without the A3D/A2D factor gives <A2D>_{beta gamma} (the A2D cancels
# the 1/A2D from I_phi), handy for forming the ratio <A3D>/<A2D>.
function integrandA2D(kaz, kbz, u, v; e = espec_odd)
    rhoA2 = 1 - kaz^2
    a2s = a2Dsqrt(rhoA2, u, v)
    a2s > 0 || return 0.0
    eb = e(rhoA2 * (u + v)^2, kbz)
    eg = e(rhoA2 * (u - v)^2, kaz + kbz)
    return rhoA2^2 * (u + v) * (u - v) * eb * eg
end

# ------------------------------------------------------------------------------------
# Numerical integration over (u, v, kbz) with the kinematic-box regulators (h, d).
# u in [1/2+10^-d, 10^h], v in [-1/2+10^-d, 1/2-10^-d], and kbz split around its two
# branch-point singularities kbz = 0 and kbz = -kaz.
# ------------------------------------------------------------------------------------
function _kbz_segments(kaz, h, d)
    if kaz < 0
        return ((-10.0^h, -10.0^(-d)),
                (10.0^(-d), -kaz - 10.0^(-d)),
                (-kaz + 10.0^(-d), 10.0^h))
    else
        return ((-10.0^h, -kaz - 10.0^(-d)),
                (-kaz + 10.0^(-d), -10.0^(-d)),
                (10.0^(-d), 10.0^h))
    end
end

function avg_bg(integrand, kaz, h, d, aTol, maxEvals; e = espec_odd)
    ulo, uhi = 0.5 + 10.0^(-d), 10.0^h
    vlo, vhi = -0.5 + 10.0^(-d), 0.5 - 10.0^(-d)
    f(x) = integrand(kaz, x[3], x[1], x[2]; e = e)
    val = 0.0
    err2 = 0.0
    for (klo, khi) in _kbz_segments(kaz, h, d)
        I, E = hcubature(f, [ulo, vlo, klo], [uhi, vhi, khi]; atol = aTol, maxevals = maxEvals)
        val += I
        err2 += E^2
    end
    return val, sqrt(err2)
end

avgA3D_bg(kaz, h, d, aTol, maxEvals; e = espec_odd) = avg_bg(integrandA3D, kaz, h, d, aTol, maxEvals; e = e)
avgA2D_bg(kaz, h, d, aTol, maxEvals; e = espec_odd) = avg_bg(integrandA2D, kaz, h, d, aTol, maxEvals; e = e)

# ------------------------------------------------------------------------------------
# <A3D>_{beta gamma} as a function of k_alpha^z, with the paper's final regulators
# h = 6, d = 5. Switch `spectrum` to compare distributions.
# ------------------------------------------------------------------------------------
spectrum = espec_odd
kazSpc = collect(0.001:0.01:1)
A3D_vec = zeros(Float64, length(kazSpc))
A3D_err = zeros(Float64, length(kazSpc))
A2D_vec = zeros(Float64, length(kazSpc))
A2D_err = zeros(Float64, length(kazSpc))

Threads.@threads for i in eachindex(kazSpc)
    A3D_vec[i], A3D_err[i] = avgA3D_bg(kazSpc[i], 6, 5, 1e-4, Int(1e8); e = spectrum)
    A2D_vec[i], A2D_err[i] = avgA2D_bg(kazSpc[i], 6, 5, 1e-4, Int(1e8); e = spectrum)
end

# --- Optional: plot and save ---------------------------------------------------------
# plot(kazSpc, A3D_vec, xlabel = L"k_\alpha^z", ylabel = L"\langle A_{3D} \rangle_{\beta\gamma}",
#      title = "3D area averaged over p, q")
# CSV.write("avgA3D_vs_k.csv", DataFrame(kaz = kazSpc, A3D = A3D_vec, A3D_err = A3D_err,
#                                        A2D = A2D_vec, A2D_err = A2D_err))

# To later average over the alpha leg (weight e_alpha = |kaz|^{-1/2}, N = 2):
#   ratio = sum(@. (A3D_vec / A2D_vec) * kazSpc^(-1/2)) * step(kazSpc) / 2
