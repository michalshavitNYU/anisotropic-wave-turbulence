using DrWatson
@quickactivate "oddWaves"
using InteractiveUtils, DataFrames, CSV, Statistics, Roots
using LaTeXStrings, HCubature, Plots

default(fontfamily="Computer Modern",
        linewidth=1, framestyle=:box, label=nothing)


########################## with rho = sqrt(1-kaz^2) ##########################

function integrandA2D_kaz(kaz, kbz, u, v )
    rhoA = sqrt(1-kaz.^2);
    Integrand = rhoA.^4 * (u+v) * (u-v) * (abs.(kaz))^(-1/2) * (abs.(kbz))^(-1/2) * (abs.(kaz .+ kbz))^(-1/2) *
    (rhoA.^2*(u+v).^2 + kbz.^2).^(-5/4) * (rhoA.^2*(u-v).^2 + (kaz+kbz).^2).^(-5/4) ./ 2;

    return Integrand
end

function integrandA3D_kaz(kaz, kbz, u, v )
        rhoA = sqrt(1-kaz.^2);
        a2Dsqrt = 4*(u+v).^2*(u-v).^2 - (2*u^2 + 2*v^2 - 1).^2;
        a3Dsqrt = 4*(rhoA.^2 * (u+v).^2 + kbz.^2)*(rhoA.^2 * (u-v).^2 + (kaz+kbz).^2) - 
        (rhoA.^2*(2*u^2+2*v^2-1) + kbz.^2 + (kaz+kbz).^2).^2;

        
        if a2Dsqrt>0 && a3Dsqrt> 0
                a2D = rhoA.^2 * sqrt(a2Dsqrt) ./ 4 ;
                a3D = sqrt(a3Dsqrt) ./ 4;

                Integrand = a3D * rhoA.^4 * (u+v) * (u-v) * (abs.(kaz))^(-1/2) * (abs.(kbz))^(-1/2) * (abs.(kaz .+ kbz))^(-1/2) *
                (rhoA.^2*(u+v).^2 + kbz.^2).^(-5/4) * (rhoA.^2*(u-v).^2 + (kaz+kbz).^2).^(-5/4) ./ (2*a2D);

                return Integrand
        else
                return 0
        end
end

uSpc = collect(1/2+1e-3: 1e-2 :100);
vSpc = collect(-1/2+1e-3 : 1e-2: 1/2-1e-3);
a2DIntegrand_mat = zeros(Float64, length(uSpc), length(vSpc));
a3DIntegrand_mat = zeros(Float64, length(uSpc), length(vSpc));

Threads.@threads for uInd in eachindex(uSpc)
        for vInd in eachindex(vSpc)
                a2DIntegrand_mat[uInd, vInd] = integrandA2D_kaz(0.1, 0.2, uSpc[uInd], vSpc[vInd] );
                a3DIntegrand_mat[uInd, vInd] = integrandA3D_kaz(0.1, 0.2, uSpc[uInd], vSpc[vInd] )
        end
end

# heatmap(a2DIntegrand_mat)
heatmap(a3DIntegrand_mat)
heatmap(uSpc, vSpc, transpose(a3DIntegrand_mat), xlabel="u", ylabel="v", clims=(0, 50) )

heatmap(uSpc, vSpc, transpose(a2DIntegrand_mat), xlabel="u", ylabel="v", clims=(0, 50) )


function a2D_NumInt_kaz(kaz, h, d, aTol, maxEVals) 
        if kaz < 0 
                ## syntax: integrandA2D_kaz(kaz, kbz, u, v )
                int, int_err = hcubature(x -> integrandA2D_kaz.(kaz, x[4], x[2], x[3]), [10. ^(-d), 1/2+10. ^(-d) , -1/2+10. ^(-d), -10^h], [1, 10^h, 1/2 -10. ^(-d),-10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int2, int_err2 = hcubature(x -> integrandA2D_kaz.(kaz, x[4], x[2], x[3]), [10. ^(-d), 1/2+10. ^(-d) , -1/2+10. ^(-d), 10. ^(-d)], [1, 10^h, 1/2 -10. ^(-d),-kaz- 10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int3, int_err3 = hcubature(x -> integrandA2D_kaz.(kaz, x[4], x[2], x[3]), [10. ^(-d), 1/2+10. ^(-d) , -1/2+10. ^(-d), -kaz+ 10. ^(-d)], [1, 10^h, 1/2 -10. ^(-d),10^h], atol=aTol, maxevals=maxEVals) 
        else
                int, int_err = hcubature(x -> integrandA2D_kaz.(kaz, x[4], x[2], x[3]), [10. ^(-d), 1/2+10. ^(-d) , -1/2+10. ^(-d), -10^h], [1, 10^h, 1/2 -10. ^(-d),-kaz-10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int2, int_err2 = hcubature(x -> integrandA2D_kaz.(kaz, x[4], x[2], x[3]), [10. ^(-d), 1/2+10. ^(-d) , -1/2+10. ^(-d), -kaz+10. ^(-d)], [1, 10^h, 1/2 -10. ^(-d),10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int3, int_err3 = hcubature(x -> integrandA2D_kaz.(kaz, x[4], x[2], x[3]), [10. ^(-d), 1/2+10. ^(-d) , -1/2+10. ^(-d), 10. ^(-d)], [1, 10^h, 1/2 -10. ^(-d),10^h], atol=aTol, maxevals=maxEVals) 
        end


    return int.+int2.+int3, sqrt(int_err.^2 .+ int_err2.^2 .+ int_err3.^2)
end

function a3D_NumInt_kaz(kaz, h, d, aTol, maxEVals) 
        if kaz < 0 
                int, int_err = hcubature(x -> integrandA3D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -10^h], [10^h, 1/2 -10. ^(-d),-10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int2, int_err2 = hcubature(x -> integrandA3D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), 10. ^(-d)], [10^h, 1/2 -10. ^(-d),-kaz- 10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int3, int_err3 = hcubature(x -> integrandA3D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -kaz+ 10. ^(-d)], [10^h, 1/2 -10. ^(-d),10^h], atol=aTol, maxevals=maxEVals) 
        else
                int, int_err = hcubature(x -> integrandA3D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -10^h], [10^h, 1/2 -10. ^(-d),-kaz-10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int2, int_err2 = hcubature(x -> integrandA3D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -kaz+10. ^(-d)], [10^h, 1/2 -10. ^(-d),10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int3, int_err3 = hcubature(x -> integrandA3D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), 10. ^(-d)], [10^h, 1/2 -10. ^(-d),10^h], atol=aTol, maxevals=maxEVals) 
        end


    return int.+int2.+int3, sqrt(int_err.^2 .+ int_err2.^2 .+ int_err3.^2)
end

kazSpc = collect(0.001:0.01:1);
integral2D_mat = zeros(Float64, length(kazSpc));
integral2D_err_mat = zeros(Float64, length(kazSpc));
integral3D_mat = zeros(Float64, length(kazSpc));
integral3D_err_mat = zeros(Float64, length(kazSpc));

## syntax a3D_NumInt_kaz(kaz, h, d, aTol, maxEVals) 

Threads.@threads for kazInd in eachindex(kazSpc)
        integral2D_mat[kazInd],integral2D_err_mat[kazInd] = a2D_NumInt_kaz.(kazSpc[kazInd], 2 , 6, 1e-4, Int(1e8)) 
        integral3D_mat[kazInd],integral3D_err_mat[kazInd] = a3D_NumInt_kaz.(kazSpc[kazInd], 2 , 6, 1e-4, Int(1e8)) 
end
integral2DPlot = plot(kazSpc, integral2D_mat, title=L"\langle A_{2D} \rangle", xlabel=L"k_\alpha^z")
savefig(integral2DPlot, plotsdir("triangleArea", "integral2D_vs_kza_v1.pdf"))

integral3DPlot = plot(kazSpc, integral3D_mat, title=L"\langle A_{3D} \rangle", xlabel=L"k_\alpha^z")
savefig(integral3DPlot, plotsdir("triangleArea", "integral3D_vs_kza_v1.pdf"))

## synatx: a2D_NumInt_kaz(kaz, h, d, aTol, maxEVals)     

function integrandAvgTheta(kaz, h, d, aTol, maxEVals)
        (a3D_NumInt_kaz(kaz, h, d, aTol, Int(maxEVals))[1])*kaz^(-1/2)./ (2*a2D_NumInt_kaz(kaz, h, d, aTol, Int(maxEVals))[1])
end


kazSpc = collect(0.001:0.01:1);
intRatio_vec = zeros(Float64, length(kazSpc));
# intRatio_err_vec = zeros(Float64, length(kazSpc));


Threads.@threads for kazInd in eachindex(kazSpc)
        intRatio_vec[kazInd] = integrandAvgTheta.(kazSpc[kazInd], 2, 6, 1e-5, 1e8) 
end
ratioIntegrandPlot = plot(kazSpc, intRatio_vec, title=L"\frac{\langle A_{3D} \rangle}{\langle A_{2D} \rangle}", xlabel=L"k_\alpha^z")
savefig(ratioIntegrandPlot, plotsdir("triangleArea", "ratioIntegrand_vs_kza_1e-5.pdf"))

function integralAvgTheta(h, d, aTol, maxEVals)
        hcubature(x -> integrandAvgTheta(x[1], h, d, aTol, maxEVals), [10. ^(-d)], [1], atol=aTol, maxevals=Int(maxEVals))[1]
end

hSpc = collect(0:0.5:2);
dSpc = collect(2:4);
intRatio_hstest_mat = zeros(Float64, length(hSpc), length(dSpc));

Threads.@threads for hInd in eachindex(hSpc)
                        for dInd in eachindex(dSpc)
                                @time intRatio_hstest_mat[hInd, dInd] = integralAvgTheta(hSpc[hInd], dSpc[dInd], 1e-4, 1e8)
                                println("Done for h=$(hSpc[hInd]), d=$(dSpc[dInd])")
                        end
end

#### Plotting the results
a3overA2_hdTest=  plot(xlabel=L"h", ylabel=L" \langle \frac{\langle A_{3D} \rangle_{\beta \gamma}}{\langle A_{2D} \rangle_{\beta \gamma}} \rangle_{\alpha}", titlefontsize=18, 
    guidefontsize=16, 
    tickfontsize=14, 
    legendfontsize=12) 
for dInd in eachindex(dSpc)
    plot!(a3overA2_hdTest, hSpc, intRatio_hstest_mat[:,dInd], label=("d=$(dSpc[dInd])"), legend=:bottom)
end
plot(a3overA2_hdTest)

savefig(a3overA2_hdTest, plotsdir("triangleArea", "a3OverA2_avgTheta_hdTest_v1.pdf"))

### I suspect h=0.5 is enough. (h=2 probably misses information along the way)
Threads.nthreads()

hVal = 1;
dSpc2 = collect(2:10); 
intRatio_dTest_vec = zeros(Float64, length(dSpc2));

Threads.@threads for dInd2 in eachindex(dSpc2)
                        @time intRatio_dTest_vec[dInd2] = integralAvgTheta(hVal, dSpc2[dInd2], 1e-5, 1e8)
                        println("Done for d=$(dSpc2[dInd2])")
end

#### Plotting the results
a3overA2_dTest=  plot(xlabel="d regulator", ylabel= L"\frac{A_{3D}}{A_{2D}}",# ylabel=L" \langle \frac{\langle A_{3D} \rangle_{\beta \gamma}}{\langle A_{2D} \rangle_{\beta \gamma}} \rangle_{\alpha}", 
    titlefontsize=18, 
    guidefontsize=16, 
    tickfontsize=14, 
    legendfontsize=12) 
plot!(a3overA2_dTest, dSpc2, intRatio_dTest_vec)

savefig(a3overA2_dTest, plotsdir("triangleArea", "a3OverA2_avgTheta_hdTest_h1_v1.pdf"))

##### Exporting the data into a CSV format, via dataframe:
areaRatioDF = DataFrame(dSpc=dSpc2, ratio=intRatio_dTest_vec); 
CSV.write(datadir("triangleArea", "areaRatio_v1"), areaRatioDF);
### Calling the dataframe from csv file (when needed):
# areaRatioDF_read = CSV.read(datadir("triangleArea", "areaRatio_v1"), DataFrame)




# Old stuff I haven't checked:


int, int_err = hcubature(x -> integrandAvgTheta(x[1]), [1e-5], [1], atol=1e-5, maxevals=Int(1e8)) 
println("The average of A3D/A2D over theta yields: $(int)")


#### Checking highest precision... might take long!

function integrandAvgTheta_precise(kaz, aTol)
        ## syntax: aND_NumInt_kaz(kaz, h, d, aTol, maxEVals) 
        (a3D_NumInt_kaz(kaz, 2, 7, aTol, Int(1e8))[1])*kaz^(-1/2)./ (2*a2D_NumInt_kaz(kaz, 2, 7, aTol, Int(1e8))[1])
end



function integralAvgTheta(aTol)
        int1, int_err1 = hcubature(x -> integrandAvgTheta_precise(x[1],aTol), [1e-7], [0.01], atol=aTol, maxevals=Int(1e8)) 
        int2, int_err2 = hcubature(x -> integrandAvgTheta_precise(x[1],aTol), [0.01], [1], atol=aTol, maxevals=Int(1e8)) 

        return int1.+int2, sqrt(int_err1.^2 .+ int_err2.^2)
end

@time integralAvgTheta(1e-4)

d3Spc = collect(3:6);
intRatio_vec3 = zeros(Float64, length(d3Spc));

Threads.@threads for d3Ind in eachindex(d3Spc)
        @time intRatio_vec3[d3Ind] = integralAvgTheta(10. ^(-d3Spc[d3Ind]))
        println("Done for d3=$(d3Spc[d3Ind])")
end

ratioIntegral_hdTest_aTol = plot(d3Spc, intRatio_vec3, title="Sampling with increasing precision", xlabel="d for aTol")
savefig(ratioIntegral_hdTest_aTol, plotsdir("triangleArea", "ratioIntegral_vs_aTol_d7.pdf"))



####### averaging over the ratio between the two

function a2Dsqrt(u,v)
        4*(u+v) .^2*(u-v) .^2 - (2*u^2 + 2*v^2 - 1) .^2;
end

function a3Dsqrt(rhoA, kaz, kbz, u, v)
        4*(rhoA .^2 * (u+v) .^2 + kbz.^2)*(rhoA .^2 * (u-v) .^2 + (kaz+kbz) .^2) - 
        (rhoA .^2*(2*u^2+2*v^2-1) + kbz .^2 + (kaz+kbz) .^2).^2;
end


function integrandA3DoA2D_kaz(rhoA, kaz, kbz, u, v )
        
        if a2Dsqrt(u,v) > 0 && a3Dsqrt(rhoA, kaz, kbz, u, v) > 0
                a2D = rhoA .^2 * sqrt(a2Dsqrt(u,v)) ./4 ;
                a3D = sqrt(a3Dsqrt(rhoA, kaz, kbz, u, v)) ./4;

                Integrand = a3D * rhoA.^4 * (u+v) * (u-v) * (abs.(kaz))^(-1/2) * (abs.(kbz))^(-1/2) * (abs.(kaz .+ kbz))^(-1/2) *
                (rhoA.^2*(u+v).^2 + kbz.^2).^(-5/4) * (rhoA.^2*(u-v).^2 + (kaz+kbz).^2).^(-5/4) ./ (2*a2D.^2);

                return Integrand
        else
                return 0
        end
end


integrandA3DoA2D_kaz(1, 0.1, 0.3, 0.6, 0.1)

uSpc = collect(1/2+10^(-6):1e-3:1.5);
vSpc = collect(-1/2+10^(-4):1e-2:1/2-10^(-4));

integrandA3DoA2D_mat = zeros(Float64, length(uSpc), length(vSpc));

Threads.@threads for uInd in eachindex(uSpc)
        for vInd in eachindex(vSpc)
                ## syntax: integrandA2D(rhoA, kaz, kbz, u, v )
                integrandA3DoA2D_mat[uInd, vInd] = integrandA3DoA2D_kaz(1, 0.2, 0.4, uSpc[uInd], vSpc[vInd] )
        end
end

heatmap(uSpc, vSpc, transpose(integrandA3DoA2D_mat), xlabel="u", ylabel="v", clims=(0, 10))


function a3DoA2D_NumInt_kaz(kaz, h, d, aTol, maxEVals) 
        if kaz < 0 
                int, int_err = hcubature(x -> integrandA3DoA2D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -10^h], [10^h, 1/2 -10. ^(-d),-10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int2, int_err2 = hcubature(x -> integrandA3DoA2D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), 10. ^(-d)], [10^h, 1/2 -10. ^(-d),-kaz- 10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int3, int_err3 = hcubature(x -> integrandA3DoA2D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -kaz+ 10. ^(-d)], [10^h, 1/2 -10. ^(-d),10^h], atol=aTol, maxevals=maxEVals) 
        else
                int, int_err = hcubature(x -> integrandA3DoA2D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -10^h], [10^h, 1/2 -10. ^(-d),-kaz-10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int2, int_err2 = hcubature(x -> integrandA3DoA2D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), -kaz+10. ^(-d)], [10^h, 1/2 -10. ^(-d),10. ^(-d)], atol=aTol, maxevals=maxEVals) 
                int3, int_err3 = hcubature(x -> integrandA3DoA2D_kaz.(kaz, x[3], x[1],x[2]), [1/2+10. ^(-d) , -1/2+10. ^(-d), 10. ^(-d)], [10^h, 1/2 -10. ^(-d),10^h], atol=aTol, maxevals=maxEVals) 
        end


    return int.+int2.+int3, sqrt(int_err.^2 .+ int_err2.^2 .+ int_err3.^2)
end


kazSpc = collect(1e-5:1e-2:1);
a3DoA2D_vec = zeros(Float64, length(kazSpc));
a3DoA2D_err_vec = zeros(Float64, length(kazSpc));


Threads.@threads for kazInd in eachindex(kazSpc)
        ## syntax: a3DoA2D_NumInt_kaz(kaz, h, d, aTol, maxEVals) 
        a3DoA2D_vec[kazInd], a3DoA2D_err_vec[kazInd],  = a3DoA2D_NumInt_kaz(kazSpc[kazInd], 2, 4, 1e-4, Int(1e8))
end

a23D_ratioPlot = plot(kazSpc, a3DoA2D_vec, xlabel=L"k_\alpha^z", ylabel=L"\langle \frac{A_{3D}}{A_{2D}} \rangle_{\beta \gamma}")
ylims!(0, 1e4)

savefig(a23D_ratioPlot, plotsdir("triangleArea", "ratioIntegral_v3_vs_kza.pdf"))




