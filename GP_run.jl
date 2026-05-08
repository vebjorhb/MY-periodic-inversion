using Plots
using LaTeXStrings
using Statistics

include("GP_reference.jl")
include("GP_kohn_sham_inversion.jl")
include("GP_potentials.jl")

setup_threading(; n_blas=1)

function setup_plots()
    # Setup environment for making automated plots
    ENV["GKS_ENCODING"] = "utf8"
    ENV["GKSwstype"]    = "100"
    ENV["PLOTS_TEST"]   = "true"

    gr()
end

function plot_potential(res, samples, vref, a; ε_last=0,
                        lims=(-Inf, Inf), shift=nothing, arg=nothing)
    x = vec(first.(DFTK.r_vectors_cart(res.scfres.basis))) ./ a
    
    vshift = []
    if shift == "max"
         vshift = [maximum(vref) - maximum(v) for v in res.vs]
    elseif shift == "min"
        vshift = [minimum(vref) - minimum(v) for v in res.vs]
    elseif shift == "mean"
        vshift = [mean(vref) - mean(v) for v in res.vs]
    else
        vshift = 0
    end
    p = plot(xlabel=L"r/a", ylabel=L"$v_g(r)$")

    snapshots = []
    for i in 1:length(samples)
        α = i/length(samples)
        if ε_last > 0 && samples[i] < ε_last
            continue
        elseif samples[i] > maximum(res.εs)
            continue
        end
        ind = argmin(abs.(res.εs .- samples[i]))
        push!(snapshots, (ind, α))
    end
    
    for (i, α) in snapshots
        expon  = floor(Int, log10(res.εs[i]))               
        prefac = round(10^(log10(res.εs[i]) - expon); digits = 1)
        label =  LaTeXString(raw"$\varepsilon=" * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")
        plot!(p, x,  res.vs[i][:,1,1] .+ vshift[i]; 
            c=cgrad(:speed)[α], label)
    end

    plot!(p, x, vref[:,1,1]; label=L"$v_\textrm{ref}$", ls=:dash, lw=2, c=:black)
    ylims!(p, lims...)
    
    p
end

function plot_density_extrapolation_norm(ref, inv; 
                                            errorlims = nothing,
                                            proxfitrange  = fill(nothing,2),
                                            extrafitrange = fill(nothing,2))
    p = plot(  ;xlabel=L"ε", xscale=:log, yscale=:log, xflip=true, 
                ylabel=L"‖ρ^ε - ρ‖_{H^{-1}}",
                xticks=10.0 .^ ( 0:-1:-9), yticks=10.0 .^ ( -1:-1:-8))

    extra = extrapolate_density(inv.εs, inv.ρs, inv.ρref, inv.scfres.basis; step=2)

    plot!(p, inv.εs, inv.hm1_norm_ρdiff;     m=:x, lw=2, label=L"‖ρ^ε - ρ‖")
    plot!(p, extra.ε_1st, extra.ρ_1st_norms; m=:x, lw=2, label=L"‖ρ_{\mathrm{R}1}^ε - ρ‖")
    plot!(p, extra.ε_2nd, extra.ρ_2nd_norms; m=:x, lw=2, label=L"‖ρ_{\mathrm{R}2}^ε - ρ‖")

    proxstart = findfirst(inv.εs .< something(proxfitrange[1], 1e-4))
    proxstop  = findlast(inv.εs .> something(proxfitrange[2], 1e-8))
    
    linfit = linear_fit(inv.εs[proxstart:proxstop], inv.hm1_norm_ρdiff[proxstart:proxstop])
    println("Linear fit: ", round(linfit[2], sigdigits=3), "*ε")
    linear(ε)    = linfit[2] *  ε

    extrastart = findfirst(extra.ε_1st .< something(extrafitrange[1], 1e-4))
    extrastop  = findlast(extra.ε_1st .> something(extrafitrange[2], 1e-8))
    
    if extrastop > length(extra.ε_1st) && extrastart > 1
        extrastop = length(extra.ε_1st)
        println("Note: The end criterion for fitting to first order extrapolation changed to ε = ", extra.ε_1st[extrastop])
    end

    if extrastop > length(extra.ε_2nd) && extrastart > 1
        extrastop = length(extra.ε_2nd)
        println("Note: The end criterion for fitting to second order extrapoloation changed to ε = ", extra.ε_2nd[extrastop])
    end

    powfit1 = power_fit(extra.ε_1st[extrastart:extrastop], extra.ρ_1st_norms[extrastart:extrastop])
    fit_1st(ε) = powfit1[1] * ε^powfit1[2]
    println("Power fit to 1st order Richardson extraploation: ", round(powfit1[1], sigdigits=3), "*ε^{", round(powfit1[2], sigdigits=4), "}\n")

    powfit2 = power_fit(extra.ε_2nd[extrastart:extrastop], extra.ρ_2nd_norms[extrastart:extrastop])
    fit_2nd(ε) = powfit2[1] * ε^powfit2[2]
    println("Power fit to 2nd order Richardson extraploation: ", round(powfit2[1], sigdigits=3), "*ε^{", round(powfit2[2], sigdigits=4), "}\n")

    plot!(p, inv.εs, linear.(inv.εs),   ls=:dash, lw=2, 
            label=LaTeXString(raw"$ρ^ε ≈ ρ + ερ'$"))
    plot!(p, inv.εs, fit_1st.(inv.εs),  ls=:dash, lw=2, 
            label=LaTeXString(raw"$ρ_{\mathrm{R}1}^ε ≈ ρ + ε^{"* string(round(powfit1[2],sigdigits=3))*raw"} ρ''$"))
    plot!(p, inv.εs, fit_2nd.(inv.εs),  ls=:dash, lw=2, 
            label=LaTeXString(raw"$ρ_{\mathrm{R}2}^ε ≈ ρ + ε^{"* string(round(powfit2[2],sigdigits=3))*raw"} ρ'''$"))

    p
end

function plot_potential_extrapolation_norm( inv;
                                            shift="max",
                                            step=1, orders=[1], 
                                            errorlims = nothing,
                                            εlims = nothing)
    vshift = []
    if shift == "max"
         vshift = [maximum(inv.vref) - maximum(v) for v in inv.vs]
    elseif shift == "min"
        vshift = [minimum(inv.vref) - minimum(v) for v in inv.vs]
    elseif shift == "mean"
        vshift = [mean(inv.vref) - mean(v) for v in inv.vs]
    else
        vshift = 0
    end

    p = plot(  ;xlabel=L"ε", xscale=:log, yscale=:log, xflip=true, 
                ylabel=L"‖v^ε - v_{\mathrm{ref}}‖_{H^{1}}",
                xticks=10.0 .^ ( 0:-1:-9), yticks=10.0 .^ ( 1:-1:-4))
    for order in orders
        extra = extrapolate_potential(inv; step, order, shift)
        plot!(p, extra.εs, extra.v_norms; m=:x, lw=1, label=L"v^ε ≈ J(∂^+_ερ^ε)" * " $(order+1)-point")
    end

    vnorm = [norm_h1(inv.basis, inv.vs[i] .- inv.vref .+ vshift[i]) for i in 1:length(inv.vs)]
    plot!(p, inv.εs, vnorm; ls=:dash, lw=2, c=:black, label="reference")
   
    errorlims = something(errorlims, (1e-4, 1e1))
    if εlims !== nothing
        xlims!(p, εlims)
    end
    ylims!(p, errorlims) 
    p
end

function plot_potential_extrapolation_path( inv, samples;
                                            shift="min",
                                            step=1, order=1, 
                                            errorlims = nothing,
                                            εlims = nothing)
    vshift = []
    if shift == "max"
         vshift = [maximum(inv.vref) - maximum(v) for v in inv.vs]
    elseif shift == "min"
        vshift = [minimum(inv.vref) - minimum(v) for v in inv.vs]
    elseif shift == "mean"
        vshift = [mean(inv.vref) - mean(v) for v in inv.vs]
    else
        vshift = 0
    end
    
    p = plot(ylabel=L"$v_{\textrm{xc}}(\mathbf{r})$")
    extra = extrapolate_potential(inv; step, order, shift)
    x = vec(first.(DFTK.r_vectors_cart(inv.basis)))

    snapshots = []
    for i in 1:length(samples)
        α = i/length(samples)
        if ε_last > 0 && samples[i] < ε_last
            continue
        elseif samples[i] > maximum(inv.εs)
            continue
        end
        ind = argmin(abs.(inv.εs .- samples[i]))
        push!(snapshots, (ind, α))
    end

    for (i, α) in snapshots
        expon  = floor(Int, log10(inv.εs[i]))                     
        prefac = round(10^(log10(inv.εs[i]) - expon); digits = 1)
        label =  LaTeXString(raw"$\varepsilon=" * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")
        plot!(p, x, extra.vs[i-order*step] .- vshift; 
            c=cgrad(:speed)[α], label)
    end

    plot!(p, x, inv.vref .- vshift; 
            label=L"$v_\textrm{xc}$", ls=:dash, c=:black)
    ylims!(p, lims...)

end



function main();
    setup_plots()

    a = 10.0
    C = 1.0
    α = 2.0
    r = 2
    Ecut = 3000
    n_electrons = 1
    εs=10.0 .^ (1:-0.125:-8.25)
    δ = 1e-3
    reftol = 1e-14 
    invtol = 1e-8
    maxiter = 10_000
    ref1 = run_reference("reference_GP_1d_$(Ecut)"; 
                            C, α, a, Ecut, n_electrons,
                            pot = x -> optical_lattice(x; a, v0=1,r), 
                            tol=reftol, 
                            method=:self_consistent_field);
    vref1 = extract_GPE_potential(ref1.basis, ref1.ρ)
    inv5 = kohn_sham_inversion(ref1.ρ; εs=10.0 .^(1:-0.125:-4),
                                    C, α, a, Ecut, n_electrons,
                                    pot = x -> optical_lattice(x; a, v0=1,r),
                                    ρtol=0, tol=invtol, ftol=1e-12,
                                    δ, maxiter,
                                    method=:self_consistent_field)
    inv1 = kohn_sham_inversion(ref1.ρ; εs,
                                    C, α, a, Ecut, n_electrons,
                                    pot = x -> optical_lattice(x; a, v0=1,r),
                                    ρtol=0, tol=invtol, ftol=1e-12,
                                    δ, maxiter,
                                    method=:direct_minimization)


    λs = [1 - 0.5/k for k in 1:maxiter]
    inv2 = kohn_sham_inversion(ref1.ρ; εs, λs,
                                    C, α, a, Ecut, n_electrons,
                                    pot = x -> optical_lattice(x; a, v0=1,r),
                                    ρtol=0, tol=invtol, ftol=1e-12,
                                    δ, maxiter,
                                    method=:proximal_point_algorithm)
    inv3 = kohn_sham_inversion(ref1.ρ; εs,
                                    C, α, a, Ecut, n_electrons,
                                    pot = x -> optical_lattice(x; a, v0=1,r),
                                    ρtol=0, tol=invtol, ftol=1e-12,
                                    δ, maxiter,
                                    method=:pde_solver)
    inv4 = kohn_sham_inversion(ref1.ρ; εs,
                                    C, α, a, Ecut, n_electrons,
                                    pot = x -> optical_lattice(x; a, v0=1,r),
                                    ρtol=0, tol=invtol, ftol=1e-12,
                                    δ, maxiter,
                                    method=:newton)
    
    
    p1 = plot(  ;xlabel=L"ε", xscale=:log, yscale=:log, xflip=true, 
                ylabel=L"‖ρ^ε - ρ_\mathrm{gs}‖_{\mathcal{X}}",
                xticks=10.0 .^ (2:-1:-9), yticks=10.0 .^ (0:-1:-10))
    plot!(p1, inv1.εs, inv1.hm1_norm_ρdiff; m=:x, lw=2, label="direct minimisation")
    plot!(p1, inv2.εs, inv2.hm1_norm_ρdiff; m=:x, lw=2, label="proximal point algorithm")
    plot!(p1, inv3.εs, inv3.hm1_norm_ρdiff; m=:x, lw=2, label="PDE solver")
    plot!(p1, inv4.εs, inv4.hm1_norm_ρdiff; m=:x, lw=2, label="Newton")
    plot!(p1, inv5.εs, inv5.hm1_norm_ρdiff; m=:x, lw=2, label="SCF")

    p2 = plot(  ;xlabel=L"ε", xscale=:log, yscale=:log, xflip=true, 
                ylabel=L"‖v_g^ε - v_g‖_{\mathcal{X}^\ast}",
                xticks=10.0 .^ (2:-1:-9), yticks=10.0 .^ (1:-1:-9))

    direct_minimization_vdiff = []
    for v in inv1.vs
        diff = mean(vref1) - mean(v)
        push!(direct_minimization_vdiff, norm_h1(inv1.scfres.basis, v .+ diff.- vref1))
    end  
    
    proxpoint_alg_vdiff = []
    for v in inv2.vs
        diff = mean(vref1) - mean(v)
        push!(proxpoint_alg_vdiff, norm_h1(inv2.scfres.basis, v .+ diff.- vref1))
    end 
    pde_solver_vdiff = []
    for v in inv3.vs
        diff = mean(vref1) - mean(v)
        push!(pde_solver_vdiff, norm_h1(inv3.scfres.basis, v .+ diff.- vref1))
    end
    newton_vdiff = []
    for v in inv4.vs
        diff = mean(vref1) - mean(v)
        push!(newton_vdiff, norm_h1(inv4.scfres.basis, v .+ diff.- vref1))
    end

    scf_vdiff = []
    for v in inv5.vs
        diff = mean(vref1) - mean(v)
        push!(scf_vdiff, norm_h1(inv5.scfres.basis, v .+ diff.- vref1))
    end

    plot!(p2, inv1.εs, direct_minimization_vdiff;   m=:x, lw=2, label="direct minimisation")
    plot!(p2, inv2.εs, proxpoint_alg_vdiff;         m=:x, lw=2, label="proximal point algorithm")
    plot!(p2, inv3.εs, pde_solver_vdiff;            m=:x, lw=2, label="PDE solver")
    plot!(p2, inv4.εs, newton_vdiff;                m=:x, lw=2, label="Newton")
    plot!(p2, inv5.εs, scf_vdiff;                   m=:x, lw=2, label="SCF")

    samples = 10.0 .^ (-1:-0.5:-4)
    p3 = plot_potential(inv4, samples, vref1, a; shift="mean")

    plot!(p1, xaxis=false, size=(1000,400),
            bottom_margin=-10*Plots.PlotMeasures.px,
            xlabel=nothing,legend_column=2, legend=:bottomleft,
            background_color_legend = :transparent, foreground_color_legend = nothing)
    plot!(p2, size=(1000,400), yaxis=:log,
            top_margin = -10 * Plots.PlotMeasures.px, legend=false,
            bottom_margin = 20 * Plots.PlotMeasures.px)
    ylims!(p1, (1e-10,1e0))
    ylims!(p2, (1e-9,5e1))
    p = plot(p1, p2, layout=(2,1), left_margin=20 * Plots.PlotMeasures.px)

    plot!(p3, size=(1000,400),foreground_color_legend = nothing, background_color_legend=nothing,
            bottom_margin=10 * Plots.PlotMeasures.px, left_margin=12 * Plots.PlotMeasures.px)

    savefig(p, "plots/GP_inversion_convergence_composite.pdf")
    savefig(p3, "plots/GP_inversion_potential.pdf")
end