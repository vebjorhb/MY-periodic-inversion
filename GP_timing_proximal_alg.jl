using Plots
using LaTeXStrings
using Statistics

include("GP_reference.jl")
include("GP_potentials.jl")
include("GP_kohn_sham_inversion.jl")

function add_line_to_plot!(p, εs, ys, converged, color, label; stderr=nothing)
    if isnothing(stderr)
        plot!(p, εs, ys; marker=:none, lw=2, color=color, label=label)
    else
        plot!(p, εs, ys;
            ribbon=stderr,
            fillalpha=0.2,
            marker=:none,
            lw=2,
            color=color,
            label=label)
    end

    scatter!(p, εs[converged], ys[converged];
        m=:o, ms=3, markerstrokewidth=1,
        color=color, label="")

    scatter!(p, εs[.!converged], ys[.!converged];
        m=:x, ms=4, markerstrokewidth=2,
        color=color, label="")
end


function run_avg(ρ; εs, nrep=5, kwargs...)
    nε = length(εs)
    runtimes = zeros(nrep, nε)
    converged = trues(nε)

    # Perform one run to initialize any compilation overhead.
    # Avoids skewing timing results with effects of
    # just-in-time compilation
    inv = kohn_sham_inversion(ρ; εs, kwargs...)

    for rep in 1:nrep
        inv = kohn_sham_inversion(ρ; εs, kwargs...)
        infos = inv.info

        for i in 1:nε
            runtimes[rep, i] = infos[i].runtime_ns * 1e-9
            converged[i] &= infos[i].converged
        end
    end

    avg_runtime = vec(mean(runtimes, dims=1))
    std_runtime = vec(std(runtimes, dims=1))
    stderr_runtime = std_runtime ./ sqrt(nrep)

    return avg_runtime, stderr_runtime, converged
end

function main()
    ENV["GKS_ENCODING"] = "utf8"
    ENV["GKSwstype"]    = "100"
    ENV["PLOTS_TEST"]   = "true"
    gr()

    a = 10.0
    C = 1.0
    α = 2.0
    r = 2
    Ecut = 3000
    n_electrons = 1

    εs = 10.0 .^ (1:-0.125:-6)
    nrep = 5
    ref = run_reference("ref"; C, α, a, Ecut, n_electrons,
                        pot = x -> optical_lattice(x; a, v0=1, r),
                        tol=1e-12)
    common_kwargs = (C=C, α=α, a=a, Ecut=Ecut, n_electrons=n_electrons,
                    pot = x -> optical_lattice(x; a, v0=1, r),
                    tol=1e-8, ftol=1e-12, δ=1e-3, maxiter=10_000)

    p = plot(; xlabel=L"\varepsilon", xscale=:log, yscale=:log, xflip=true,
                ylabel="Mean runtime [s]", size=(1000,400),
                xticks=10.0 .^ (2:-1:-9), yticks=10.0 .^ (5:-1:-9),
                foreground_color_legend = nothing,
                background_color_legend = nothing)
    λseqs = [([2 - 1/k for k in 1:10_000], L"\lambda_k = 2 - \frac{1}{k}"),
            ([1 - 0.5/k for k in 1:10_000], L"\lambda_k = 1 - \frac{1}{2k}"),
            (1.0, L"\lambda = 1"),
            (1e-1, L"\lambda = 10^{-1}")]

    color_idx = 1
    for (λs, label) in λseqs
        avg_rt, stderr_rt, conv = run_avg(ref.ρ; εs, λs, nrep,
                            method=:proximal_point_algorithm,
                            common_kwargs...)
        add_line_to_plot!(p, εs, avg_rt, conv, color_idx, label, stderr=stderr_rt)
        color_idx += 1
    end

    methods = [ (:pde_solver, "PDE solver"),
                (:newton, "Newton"),
                (:direct_minimization, "Direct minimization")]

    for (method, label) in methods
        avg_rt, stderr_rt, conv = run_avg(ref.ρ; εs, method, nrep,common_kwargs...)
        add_line_to_plot!(p, εs, avg_rt, conv, color_idx, label, stderr=stderr_rt)
        color_idx += 1
    end

    plot!(p, legend_columns=2, bottom_margin=15Plots.PlotMeasures.px,
            left_margin=15Plots.PlotMeasures.px)
    savefig(p, "plots/GP_1d_prox_alg_runtime.pdf")
end

# Only run when this file is executed as a script
if !isnothing(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end