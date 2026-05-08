using LinearAlgebra
using Plots
using LaTeXStrings
using JSON3
using DFTK

include("term_dualmap.jl")

default(guidefontsize = 14, 
        tickfontsize = 12, 
        legendfontsize = 12, 
        titlefontsize = 16,
        fontfamily = "Computer Modern")

function setup_plots()
    # Setup environment for making automated plots
    ENV["GKS_ENCODING"] = "utf8"
    ENV["GKSwstype"]    = "100"
    ENV["PLOTS_TEST"]   = "true"

    gr()
    # default(size=tuple(Int.(ceil.(0.75 .* [600, 400]))...),
    #         guidefontsize=10, grid=false)
end

function load_path(pathfile)
    @assert endswith(pathfile, "_path.json")
    open(JSON3.read, pathfile)
end

function add_path!(p, pathdata::AbstractDict; arg=nothing)
    add_path!(p, pathdata["pathlength"], pathdata["branch_starts"],
              pathdata["atoms"], pathdata["atom_symbols"], arg=arg)
end
function add_path!(p, pathlength, branch_starts, atoms, atom_symbols; arg=nothing)
    vline!(p, getindex.(Ref(pathlength), first.(branch_starts)), label="", c=:grey, ls=:dashdot)
    vline!(p, [pathlength[end]], label="", c=:grey, ls=:dashdot)
    vline!(p, getindex.(Ref(pathlength), atoms), label="", c=:grey, ls=:dash)

    inds = getindex.(Ref(pathlength), first.(branch_starts))
    push!(inds, pathlength[end])
    labels = ["("*branch_starts[i][2]*")" for i in range(1, size(branch_starts, 1))]
    l = length(inds)
    ticks = Float64[]
    for i in range(1,l-1)
        push!(ticks, (inds[i]+inds[i+1])/2)
    end

    if arg == "Si"
        push!(labels, raw"$O$", raw"$O'$", raw"$O''$", raw"$O$")
        for i in getindex.(Ref(pathlength), first.(branch_starts))
            push!(ticks, i)
        end
        push!(ticks, pathlength[end])
    end
    j = 1
    for i in getindex.(Ref(pathlength), atoms)
        push!(ticks, i)
        push!(labels, atom_symbols[j])
        j += 1
    end
    xticks!(p, ticks, labels)
    p
end

function is_reference(filename)
    bn, _ = splitext(basename(filename))
    startswith(bn, "reference_")
end
function is_inversion(filename)
    bn, _ = splitext(basename(filename))
    (   startswith(bn, "inversion_")
     || startswith(bn, "truncate")
     || startswith(bn, "noise"))
end

function load_convergence_data(basename; ε_last=0, relative_error=false)
    @assert is_inversion(basename)
    @assert isfile(basename * ".json")
    data = open(JSON3.read, basename * ".json", "r")
    εs = data["inversion_εs"]
    εmask = εs .≥ ε_last
    εs = εs[εmask]

    errors_ρ_hm1    = data["inversion_errors_ρ_hm1"][εmask]
    errors_ρ_l2     = data["inversion_errors_ρ_l2"][εmask]
    errors_v_h1     = data["inversion_errors_v_h1"][εmask]
    errors_v_l2     = data["inversion_errors_v_l2"][εmask]

    referror_ρ_hm1 = get(data, "inversion_referror_ρ_hm1", 0.0)
    refnorm_ρ_hm1  = data["inversion_refnorm_ρ_hm1"]
    refnorm_ρ_l2   = data["inversion_refnorm_ρ_l2"]
    refnorm_v_h1   = data["inversion_refnorm_v_h1"]
    refnorm_v_l2   = data["inversion_refnorm_v_l2"]

    norms_ρ_hm1 = errors_ρ_hm1
    norms_ρ_l2  = errors_ρ_l2
    norms_v_h1  = errors_v_h1
    norms_v_l2  = errors_v_l2
    if relative_error
        norms_ρ_hm1     /= refnorm_ρ_hm1
        norms_ρ_l2      /= refnorm_ρ_l2
        norms_v_h1      /= refnorm_v_h1
        norms_v_l2      /= refnorm_v_l2
    end

    (; εs, errors_ρ_hm1, errors_ρ_l2, errors_v_h1, errors_v_l2,
       refnorm_ρ_hm1, refnorm_ρ_l2, refnorm_v_h1, refnorm_v_l2,
       referror_ρ_hm1,
       norms_ρ_hm1, norms_ρ_l2, norms_v_h1, norms_v_l2
   )
end

function plot_potential(basename; ε_last=0, refkey="vxc", lims=(-Inf, Inf), shift=nothing, arg=nothing)
    @assert isfile(basename * "_path.json")
    path = load_path(basename * "_path.json")
    if shift == "max"
        vshift = maximum(path["vref"]) + 1e-4
    elseif shift == "min"
        vshift = minimum(path["vref"])
    elseif shift == "mean"
        vshift = mean(path["vref"])
    else
        vshift = 0
    end
    if path["kind"] == "inversion"
        εmask = path["εs"] .≥ ε_last
        data  = (; vs=path["vs"][εmask], εs=path["εs"][εmask])

        p = plot(ylabel=L"$v_{\textrm{xc}}(\mathbf{r})$")
        snapshots = [(0, 0.7), (3, 0.6), (6, 0.5), (9, 0.4)]
        length(data.εs) > 12 && push!(snapshots, (12, 0.2))
        length(data.εs) > 15 && push!(snapshots, (15, 0.1))
        for (i, α) in reverse(snapshots)
            expon  = floor(Int, log10(data.εs[end-i]))                        
            prefac = round(10^(log10(data.εs[end-i]) - expon); digits = 1) 
            label =  LaTeXString(raw"$\varepsilon=" * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")
            plot!(p, path["pathlength"], data.vs[end-i] .- vshift; 
                c=cgrad(:speed)[α], label)
        end
        plot!(p, [], [];label=" ",c=nothing)
        plot!(p, path["pathlength"], path["vref"] .- vshift; 
            label=L"$v_\textrm{xc}$ [PBE PsP]", c=:black, ls=:dash)
    elseif path["kind"] == "reference"
        p = plot(path["pathlength"], path[refkey] .- vshift; 
                label=L"$v_\textrm{xc}$", c=:black)
    else
        error("Unknown kind")
    end
    add_path!(p, path, arg=arg)
    ylims!(p, lims...)
end

function plot_potential_error(  basename; 
                                relative_error=false, ε_last=0, 
                                errorlims=nothing, shift=nothing, 
                                arg=nothing)
    if relative_error
        errorlims = something(errorlims, (-1, 1))
    else
        errorlims = something(errorlims, (-Inf, Inf))
    end
   
    @assert isfile(basename * "_path.json")
    path = load_path(basename * "_path.json")
    @assert path["kind"] == "inversion"
    εmask = path["εs"] .≥ ε_last
    data = (; vref=path["vref"], vs=path["vs"][εmask], εs=path["εs"][εmask])
    
    if shift == "max"
        vshift = maximum(path["vref"]) + 1e-4
    elseif shift == "min"
        vshift = minimum(path["vref"])
    elseif shift == "mean"
        vshift = mean(path["vref"])
    else
        vshift = 0
    end

    v_error = [abs.(v - data.vref) for v in data.vs]
    if relative_error
        v_error = [abs.(v ./ (data.vref .- vshift)) for v in v_error]
    end

    ylabel = relative_error ? "Relative error" : "Absolute pointwise error"
    p = plot(; ylabel)
    snapshots = [(0, 0.7), (3, 0.6), (6, 0.5), (9, 0.4)]
    length(data.εs) > 8 && push!(snapshots, (8, 0.2))
    length(data.εs) > 10 && push!(snapshots, (10, 0.1))
    for (i, α) in reverse(snapshots)
        label = "ε = $(round(data.εs[end-i]; sigdigits=2))"
        plot!(p, path["pathlength"], v_error[end-i]; c=cgrad(:speed)[α], label)
    end
 
    add_path!(p, path, arg=arg)
    ylims!(p, errorlims...)
end


function plot_density_convergence(basenames; relative_error=false, ε_last=0, errorlims=nothing,
                                    labels=nothing, colors=nothing, ε_extra=false, legend_error=false,
                                    norm_ρ=:norms_ρ_hm1)
    p = plot(;  xaxis=:log, yaxis=:log, xflip=true, xlabel=L"ε")
    common = (; mark=:x, lw=1.5)
    if labels == nothing
        labels = [replace(basename, "results/inversion_" => "") for basename in basenames]
    end
    if colors == nothing
        colors = palette(:default)[1:length(basenames)] 
    end

    norm = nothing
    for (basename, label, color) in zip(basenames, labels, colors)
        @assert is_inversion(basename)
        data = load_convergence_data(basename; ε_last, relative_error)  
        if legend_error == true
            Δρ_hm1 = try
                    data.referror_ρ_hm1
                catch
                    0
                end

            if Δρ_hm1 != 0
                expon  = floor(Int, log10(Δρ_hm1))                       # Computing labels for each  
                prefac = round(10^(log10(Δρ_hm1) - expon); digits = 1)   # Δρ using the H^-1 norm 
                label = LaTeXString(label * raw", $‖Δρ‖ = " 
                        * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")
            else
                label = LaTeXString(label * raw", $‖Δρ‖ = 0$")
            end
        end

        bn, _ = splitext(basename)
        if norm_ρ == :norms_ρ_hm1 && startswith(bn, "results/truncate")
            norm = data.norms_ρ_hm1
            plot!(p; ylabel=L"‖\tilde{ρ}^ε - \tilde{ρ}_\mathrm{ref}‖_{\mathcal{X}}")
        elseif norm_ρ == :norms_ρ_hm1
            norm = data.norms_ρ_hm1
            plot!(p; ylabel=L"‖ρ^ε - ρ_\mathrm{gs}‖_{\mathcal{X}}")
        elseif norm_ρ ==:norms_ρ_l2 && startswith(bn, "results/truncate")
            norm = data.norms_ρ_l2
            plot!(p; ylabel=L"‖\tilde{ρ}^ε - \tilde{ρ}_\mathrm{ref}‖_{L^2}")
        elseif norm_ρ ==:norms_ρ_l2 
            norm = data.norms_ρ_l2
            plot!(p; ylabel=L"‖ρ^ε - ρ_\mathrm{gs}‖_{L^2}")
        else 
            @error "Unknown ρ norm: $(norm_ρ)"
        end
        plot!(p, data.εs, norm; common..., label, color)
        if ε_extra == true && isfile(basename * "_extra.json")
            extra_data = load_convergence_data(basename * "_extra"; ε_last, relative_error)
            εmask_extra = (extra_data.εs .< data.εs[end]) .& (extra_data.εs .>= ε_last)
            if any(εmask_extra)
                extra_norm = nothing
                if norm_ρ == :norms_ρ_hm1
                    extra_norm = extra_data.norms_ρ_hm1
                elseif norm_ρ ==:norms_ρ_l2
                    extra_norm = extra_data.norms_ρ_l2
                end
                plot!(p,    [data.εs[end]; extra_data.εs[εmask_extra]],
                            [norm[end]; extra_norm[εmask_extra]];
                            common..., label="", color)
            end
        end
    end
    
    if errorlims != nothing
        ylims!(p, errorlims...)
    end
    p
end

function plot_potential_convergence(basenames; relative_error=false, ε_last=0, errorlims=nothing,
                                    labels=nothing, colors=nothing, ε_extra=false, legend_error=false)
    p = plot(;  xaxis=:log, yaxis=:log, xflip=true, xlabel=L"ε", 
                ylabel=L"‖v^ε_\mathrm{xc} - v_\mathrm{xc}‖_{\mathcal{X}^\ast}")
    common = (; mark=:x, lw=1.5)
    if labels == nothing
        labels = [replace(basename, "results/inversion_" => "") for basename in basenames]
    end
    if colors == nothing
        colors = palette(:default)[1:length(basenames)] 
    end

    for (basename, label, color) in zip(basenames, labels, colors)
        @assert is_inversion(basename)
        data = load_convergence_data(basename; ε_last, relative_error)
        if legend_error == true
            Δρ_hm1 = try
                    data.referror_ρ_hm1
                catch
                    0
                end

            if Δρ_hm1 != 0
                expon  = floor(Int, log10(Δρ_hm1))                      
                prefac = round(10^(log10(Δρ_hm1) - expon); digits = 1) 
                label = LaTeXString(label * raw", $‖Δρ‖ = " 
                        * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")
            else
                label = LaTeXString(label * raw", $‖Δρ‖ = 0$")
            end
        end

        plot!(p, data.εs, data.norms_v_h1; common..., label, color)
        if ε_extra == true && isfile(basename * "_extra.json")
            extra_data = load_convergence_data(basename * "_extra"; ε_last, relative_error)
            εmask_extra = (extra_data.εs .< data.εs[end]) .& (extra_data.εs .>= ε_last)
            if any(εmask_extra)
                plot!(p,    [data.εs[end]; extra_data.εs[εmask_extra]], 
                            [data.norms_v_h1[end]; extra_data.norms_v_h1[εmask_extra]];
                             common..., label="", color)
            end
        end
    end
    
    if errorlims != nothing
        ylims!(p, errorlims...)
    end
    p
end

function plot_perturbation_analysis(basenames::AbstractVector{<:AbstractString};
                                    refname=nothing, reflabel="",
                                    labels=basenames, ε_last=0,
                                    colors=collect(1:length(basenames)),
                                    legend_error=false)
    @assert length(labels) == length(basenames)
    datas = map(basenames) do bn
        @assert isfile(bn * "_perturb.json")
        open(JSON3.read, bn * "_perturb.json", "r")
    end
    Qlabel = L"$Q_\varepsilon(\Delta \rho)$"
    Rlabel = L"$R_\varepsilon(\Delta \rho)$"
    Slabel = L"$S_\varepsilon(\Delta \rho)$"

    p_Q = plot(; yaxis=:log, xaxis=:log, xflip=true, legend=:bottomright, 
                xlabel=L"$\varepsilon$", ylabel=Qlabel)
    p_R = plot(; xaxis=:log, xflip=true, legend=:bottomleft, 
                xlabel=L"$\varepsilon$", ylabel=Rlabel)
    p_S = plot(; xaxis=:log, xflip=true, legend=:topleft, 
                xlabel= L"$\varepsilon$", ylabel=Slabel)
    p_dens = plot(; yaxis=:log, xaxis=:log, xflip=true,  
                xlabel=L"$\varepsilon$",  legend=:bottomleft,
                ylabel=L"‖\tilde{ρ}^ε - ρ_\mathrm{gs}‖_{\mathcal{X}}")
    if !isnothing(refname)
        ref = load_convergence_data(refname; ε_last)
        plot!(p_dens, ref.εs, ref.norms_ρ_hm1; c=:black, mark=:x, label=reflabel)
    end
    for (data, label, color) in zip(datas, labels, colors)
        εmask = data.εs .≥ ε_last
        if legend_error == true
            expon  = floor(Int, log10(data.Δρ_hm1))                       
            prefac = round(10^(log10(data.Δρ_hm1) - expon); digits = 1)
            label  = LaTeXString(label * raw", $‖Δρ‖ = " 
                        * string(prefac) * raw"\times 10^{" * string(expon) * raw"}$")
        end
       
        plot!(p_R, data.εs[εmask], data.Rεs[εmask]; label, color, mark=:x)
        plot!(p_S, data.εs[εmask], data.Sεs[εmask]; label, color, mark=:x)      
        plot!(p_Q, data.εs[εmask], data.Qεs[εmask]; label, color, mark=:x)
        plot!(p_dens, data.εs[εmask], data.truncation_errors_ρ_hm1[εmask];
                    label, color, mark=:x)
    end

    (; p_Q, p_R, p_S, p_dens)
end



function main()
    setup_plots()
    xticks = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    ε_last = 1e-7  
    Si_rel = plot_potential_error("results/inversion_silicon_Ecut_45_kgrid_10";
                                    ε_last, relative_error=true, errorlims=(2e-6, 1), arg="Si")
    # Potential plot in exact inversion
    Si_pot = plot_potential("results/inversion_silicon_Ecut_45_kgrid_10"; ε_last)
    plot!(Si_pot; xaxis=false, size=(1000,400), bottom_margin=-50*Plots.PlotMeasures.px,
                legend_column=4, legend=(0.1, 1.16),
                background_color_legend = :transparent, foreground_color_legend = nothing, grid=false)
    plot!(Si_rel; size=(1000,400), legend=false, yaxis=:log, yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                top_margin=40 * Plots.PlotMeasures.px, grid=false)
    Si = plot(Si_pot, Si_rel, layout=(2,1), 
                top_margin=31 * Plots.PlotMeasures.px, left_margin=20 * Plots.PlotMeasures.px)
    savefig(Si, "plots/Si_inversion_Ecut_45_kgrid_10_composite.pdf")
    
    ε_last = 1e-7 
    KCl_rel = plot_potential_error("results/inversion_kcl_Ecut_65_kgrid_15_alpha_1.0_beta_0.0_gamma_1.0"; 
                                    ε_last, relative_error=true, errorlims=(2e-6, 1))
    KCl_pot = plot_potential("results/inversion_kcl_Ecut_65_kgrid_15_alpha_1.0_beta_0.0_gamma_1.0"; ε_last)
    plot!(KCl_pot; xaxis=false, size=(1000,400), bottom_margin=-50*Plots.PlotMeasures.px,
                legend_column=4, legend=(0.1, 1.18),
                background_color_legend = :transparent, foreground_color_legend = nothing, grid=false)
    plot!(KCl_rel; size=(1000,400), legend=false, yaxis=:log, yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                top_margin=40 * Plots.PlotMeasures.px, grid=false)
    KCl = plot(KCl_pot, KCl_rel, layout=(2,1), 
                top_margin=31 * Plots.PlotMeasures.px, left_margin=20 * Plots.PlotMeasures.px)
    savefig(KCl, "plots/KCl_inversion_Ecut_65_kgrid_15_alpha_1.0_beta_0.0_gamma_1.0_composite.pdf")
    
    ε_last = 1e-7  
    GaAs_rel = plot_potential_error(   "results/inversion_gaas_Ecut_83_kgrid_17";
                                    ε_last, relative_error=true, errorlims=(2e-6, 1))
    # Potential plot in exact inversion
    GaAs_pot = plot_potential("results/inversion_gaas_Ecut_83_kgrid_17"; ε_last)
    plot!(GaAs_pot; xaxis=false, size=(1000,400), bottom_margin=-50*Plots.PlotMeasures.px,
                legend_column=4, legend=(0.1, 1.16),
                background_color_legend = :transparent, foreground_color_legend = nothing, grid=false)
    plot!(GaAs_rel; size=(1000,400), legend=false, yaxis=:log, yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                top_margin=40 * Plots.PlotMeasures.px, grid=false)
    GaAs = plot(GaAs_pot, GaAs_rel, layout=(2,1), 
                top_margin=31 * Plots.PlotMeasures.px, left_margin=20 * Plots.PlotMeasures.px)
    savefig(GaAs, "plots/inversion_gaas_Ecut_83_kgrid_17_composite.pdf")
    
    ε_last = 1e-7  
    NaCl_rel = plot_potential_error(   "results/inversion_NaCl_Ecut_83_kgrid_17";
                                    ε_last, relative_error=true, errorlims=(2e-6, 1))
    # Potential plot in exact inversion
    NaCl_pot = plot_potential("results/inversion_NaCl_Ecut_83_kgrid_17"; ε_last)
    plot!(NaCl_pot; xaxis=false, size=(1000,400), bottom_margin=-50*Plots.PlotMeasures.px,
                legend_column=4, legend=(0.1, 1.16),
                background_color_legend = :transparent, foreground_color_legend = nothing, grid=false)
    plot!(NaCl_rel; size=(1000,400), legend=false, yaxis=:log, yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                top_margin=40 * Plots.PlotMeasures.px, grid=false)
    NaCl = plot(NaCl_pot, NaCl_rel, layout=(2,1), 
                top_margin=31 * Plots.PlotMeasures.px, left_margin=20 * Plots.PlotMeasures.px)
    savefig(NaCl, "plots/inversion_NaCl_Ecut_83_kgrid_17_composite.pdf")

    # Convergence plots
    ε_last = 1e-8
    files = ["results/inversion_silicon_Ecut_45_kgrid_10",
            "results/inversion_kcl_Ecut_65_kgrid_15_alpha_1.0_beta_0.0_gamma_1.0",
            "results/inversion_gaas_Ecut_83_kgrid_17",
            "results/inversion_NaCl_Ecut_83_kgrid_17"]
    labels = ["Si", "KCl", "GaAs", "NaCl"]

    p1 = plot_density_convergence(files; ε_last, labels)
    plot!(p1; size=(1000,400), xaxis=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            bottom_margin=-10*Plots.PlotMeasures.px,
            xlabel=nothing,legend_column=2, legend=:bottomleft,
            background_color_legend = :transparent, foreground_color_legend = nothing)


    p2 = plot_potential_convergence(files; errorlims=(1e-2, 10), ε_last, labels)
    plot!(p2; size=(1000,400), legend=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1],
            top_margin = -10 * Plots.PlotMeasures.px, 
            bottom_margin = 20 * Plots.PlotMeasures.px)
    # savefig(p2, "plots/potential_convergence.pdf")
    p = plot(p1, p2, layout=(2,1), left_margin=20 * Plots.PlotMeasures.px)

    """
    Exploring using in parts the exact Hartree potential
    
    Using the guiding functionals with different α, β, γ weights
        F(ρ) = T(ρ) + α E_H(ρ) + β ⟨v_H[ρgs], ρ⟩ + γ ⟨vext, ρ⟩
    
    α = 1.0, β = 0.0, γ = 1.0  (''standard'' PRB functional)
    α = 0.0, β = 1.0, γ = 1.0  (using exact Hartree potential only)
    """
    ε_last = 1e-7
    files = ["results/inversion_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_0.0_gamma_1.0",
            "results/inversion_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0",
            "results/inversion_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0",
            "results/inversion_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_0.9",
            "results/inversion_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.1",
            "results/inversion_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_0.0",
            "results/inversion_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_0.0",
            "results/inversion_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_0.0_gamma_0.0",
            ]
    labels = [  "α = 0, β = 0, γ = 1",
                "α = 1, β = 0, γ = 1",
                "α = 0, β = 1, γ = 1",
                "α = 1, β = 0, γ = 0.9",
                "α = 1, β = 0, γ = 1.1",
                "α = 1, β = 0, γ = 0",
                "α = 0, β = 1, γ = 0",
                "α = 0, β = 0, γ = 0"
              ]
    p3 = plot_density_convergence(files; ε_last, labels)
    plot!(p3; size=(1000,400), xaxis=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            bottom_margin=-10*Plots.PlotMeasures.px,
            xlabel=nothing,legend_column=2, legend=(0.09,0.4),
            background_color_legend = :transparent, foreground_color_legend = nothing)

    p4 = plot_potential_convergence(files; errorlims=(1e-2, 100), ε_last, labels)
    plot!(p4; size=(1000,400), legend=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1],
            top_margin = -10 * Plots.PlotMeasures.px, 
            bottom_margin = 20 * Plots.PlotMeasures.px)
    p = plot(p3, p4, layout=(2,1), left_margin=20 * Plots.PlotMeasures.px)
    savefig(p, "plots/Si_functionals_convergence_composite.pdf")

    files = [   "results/inversion_kcl_Ecut_65_kgrid_15_alpha_1.0_beta_0.0_gamma_1.0",
                "results/inversion_kcl_Ecut_65_kgrid_15_alpha_0.0_beta_1.0_gamma_1.0",
                "results/inversion_kcl_Ecut_65_kgrid_15_alpha_1.0_beta_0.0_gamma_1.1",]
    labels = [  "α = 1.0, β = 0.0, γ = 1.0",
                "α = 0.0, β = 1.0, γ = 1.0",
                "α = 1.0, β = 0.0, γ = 1.1",
              ]
    p5 = plot_density_convergence(files; ε_last, labels)
    plot!(p5; size=(1000,400), xaxis=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            bottom_margin=-10*Plots.PlotMeasures.px,
            xlabel=nothing,legend_column=2, legend=(0.09,0.2),
            background_color_legend = :transparent, foreground_color_legend = nothing)
    

    p6 = plot_potential_convergence(files; errorlims=(1, 10), ε_last, labels)
    plot!(p6; size=(1000,400), legend=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1],
            top_margin = -10 * Plots.PlotMeasures.px, 
            bottom_margin = 20 * Plots.PlotMeasures.px)
    p = plot(p5, p6, layout=(2,1), left_margin=20 * Plots.PlotMeasures.px)
    savefig(p, "plots/KCl_functionals_convergence_composite.pdf")

    truncation_Si_alpha = [ "results/inversion_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0",
                            "results/truncate35_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0",
                            "results/truncate30_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0",
                            "results/truncate25_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0",
                            "results/truncate20_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0",
                            "results/truncate15_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0",
                            "results/truncate10_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0"
                          ]
    extra_Si_alpha =   ["results/truncate35_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0_extra",
                        "results/truncate30_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0_extra",
                        "results/truncate25_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0_extra",
                        "results/truncate20_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0_extra",
                        "results/truncate15_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0_extra",
                        "results/truncate10_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0_extra"

    ]
    labels_Si_alpha = [ raw"$E_{\mathrm{cut}} = 45$ Ha",
                        raw"$E_{\mathrm{cut}} = 35$ Ha",
                        raw"$E_{\mathrm{cut}} = 30$ Ha",
                        raw"$E_{\mathrm{cut}} = 25$ Ha",
                        raw"$E_{\mathrm{cut}} = 20$ Ha",
                        raw"$E_{\mathrm{cut}} = 15$ Ha",
                        raw"$E_{\mathrm{cut}} = 10$ Ha"
                        ]
    p8 = plot_potential_convergence(truncation_Si_alpha; ε_last=1e-6, errorlims=(1e-2, 10), 
                                    labels=labels_Si_alpha, ε_extra=true, legend_error=true,
                                    colors = [:black; palette(:default)[1:length(truncation_Si_alpha)-1]] )
    pQ_alpha, pR_alpha, pS_alpha, p7 = plot_perturbation_analysis(truncation_Si_alpha[2:end];
                                                                ε_last=1e-6, 
                                                                labels=labels_Si_alpha[2:end], 
                                                                refname=truncation_Si_alpha[1],
                                                                reflabel=labels_Si_alpha[1],
                                                                legend_error=false)
    plot!(p7; size=(1000,400), xaxis=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            bottom_margin=-10*Plots.PlotMeasures.px,
            xlabel=nothing,legend_column=2, legend=(0.09,0.4),
            background_color_legend = :transparent, foreground_color_legend = nothing)
    plot!(p8; size=(1000,400), legend=false,
            xticks=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            yticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1],
            top_margin = -10 * Plots.PlotMeasures.px, 
            bottom_margin = 20 * Plots.PlotMeasures.px)
    
    p = plot(p7, p8, layout=(2,1), left_margin=20 * Plots.PlotMeasures.px)
    savefig(p, "plots/Si_trunctations_composite.pdf")
    plot!(pQ_alpha; 
            size=(800,350), xticks, yticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0], 
            left_margin   = 25 * Plots.PlotMeasures.px,
            bottom_margin = 20 * Plots.PlotMeasures.px,
            background_color_legend = :transparent, foreground_color_legend = nothing)
    savefig(pQ_alpha, "plots/Si_Qε_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0.pdf")

    plot!(pS_alpha;
                bottom_margin=-10*Plots.PlotMeasures.px,
                legend_column=2, legend=(0.6, 0.25), xticks,
                background_color_legend = :transparent, foreground_color_legend = nothing)
    plot!(pR_alpha; size=(1000,400),
                xticks, background_color_legend = :transparent, foreground_color_legend = nothing,
                top_margin = -10 * Plots.PlotMeasures.px, 
                bottom_margin = 20 * Plots.PlotMeasures.px)
    pRS_alpha = plot(pS_alpha, pR_alpha, layout=(2,1), 
                left_margin=20 * Plots.PlotMeasures.px)
    savefig(pRS_alpha, "plots/Si_SεRε_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0.pdf")

    plot!(pR_alpha; 
            size=(800,350), xticks,
            left_margin   = 25 * Plots.PlotMeasures.px,
            bottom_margin = 20 * Plots.PlotMeasures.px,
            background_color_legend = :transparent, foreground_color_legend = nothing)
    plot!(pS_alpha; 
            size=(800,350), xticks,
            left_margin   = 25 * Plots.PlotMeasures.px,
            bottom_margin = 20 * Plots.PlotMeasures.px,
            background_color_legend = :transparent, foreground_color_legend = nothing)
    savefig(pR_alpha, "plots/Si_Rε_Ecut_45_kgrid_10_alpha_1.0_beta_1.0_gamma_1.0.pdf")
    savefig(pS_alpha, "plots/Si_Sε_Ecut_45_kgrid_10_alpha_1.0_beta_1.0_gamma_1.0.pdf")

    truncation_Si_beta = [  "results/inversion_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0",
                            "results/truncate35_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0",
                            "results/truncate30_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0",
                            "results/truncate25_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0",
                            "results/truncate20_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0",
                            "results/truncate15_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0",
                            "results/truncate10_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0"
                            ]
    extra_Si_beta =   [ "results/truncate30_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0_extra",
                        "results/truncate25_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0_extra",
                        "results/truncate20_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0_extra"
    ]
    labels_Si_beta = [  raw"$E_{\mathrm{cut}} = 45$ Ha",
                        raw"$E_{\mathrm{cut}} = 35$ Ha",
                        raw"$E_{\mathrm{cut}} = 30$ Ha",
                        raw"$E_{\mathrm{cut}} = 25$ Ha",
                        raw"$E_{\mathrm{cut}} = 20$ Ha",
                        raw"$E_{\mathrm{cut}} = 15$ Ha",
                        raw"$E_{\mathrm{cut}} = 10$ Ha"
                        ]
    
    p10 = plot_potential_convergence(truncation_Si_beta; ε_last=1e-6, errorlims=(1e-2, 10), 
                                    labels=labels_Si_beta, ε_extra=true, legend_error=true)
    pQ_beta, pR_beta, pS_beta, p9 = plot_perturbation_analysis( truncation_Si_beta[2:end];
                                                                ε_last=1e-6, 
                                                                labels=labels_Si_beta[2:end], 
                                                                refname=truncation_Si_beta[1], 
                                                                reflabel=labels_Si_beta[1], 
                                                                legend_error=false)
    plot!(p9; size=(1000,400), legend=:bottomleft,
            xticks, ylabel=L"‖\tilde{ρ}^ε - \tilde{ρ}_\mathrm{ref}‖_{\mathcal{X}}",
            yticks=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            left_margin   = 25 * Plots.PlotMeasures.px,
            bottom_margin = 20 * Plots.PlotMeasures.px,
            background_color_legend = :transparent, foreground_color_legend = nothing)
    plot!(p10; size=(1000,400), legend=:bottomleft,
            xticks, ylabel=L"‖\tilde{v}^ε_\mathrm{xc} - v_\mathrm{xc}‖_{\mathcal{X}^\ast}",
            yticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1],
            left_margin   = 25 * Plots.PlotMeasures.px,
            bottom_margin = 20 * Plots.PlotMeasures.px,
            background_color_legend = :transparent, foreground_color_legend = nothing)

    savefig(p9, "plots/Si_density_truncation_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0.pdf")
    savefig(p10, "plots/Si_potential_truncation_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0.pdf")
    
    plot!(pQ_beta; 
            size=(1000,400), xticks, yticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0], 
            left_margin   = 25 * Plots.PlotMeasures.px,
            bottom_margin = 20 * Plots.PlotMeasures.px,
            background_color_legend = :transparent, foreground_color_legend = nothing)
    ylims!(pQ_beta, (1e-2, 1.1))
    savefig(pQ_beta, "plots/Si_Qε_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0.pdf")

    plot!(pS_beta; xaxis=false, size=(1000,400),
                bottom_margin=-10*Plots.PlotMeasures.px,
                legend_column=2, legend=(0.6, 0.25), xticks, xlabel=nothing,
                background_color_legend = :transparent, foreground_color_legend = nothing)
    plot!(pR_beta; size=(1000,400), legend=false, xticks,
                top_margin = -10 * Plots.PlotMeasures.px, 
                bottom_margin = 20 * Plots.PlotMeasures.px)
    pRS_beta = plot(pS_beta, pR_beta, layout=(2,1), 
                    left_margin=20 * Plots.PlotMeasures.px)
    savefig(pRS_beta, "plots/Si_SεRε_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0.pdf")
end