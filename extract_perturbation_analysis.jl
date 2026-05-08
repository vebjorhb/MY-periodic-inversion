
using LinearAlgebra
using JLD2
using JSON3
using DFTK
include("kohn_sham_inversion.jl")

function is_inversion(filename)
    bn, _ = splitext(basename(filename))
    (startswith(bn, "inversion_")
     || startswith(bn, "truncate")
    )
end

function perturbfile(filename)
    @assert isfile(filename)
    bn, _ = splitext(filename)
    bn * "_perturb.json"
end

function compute_perturbation_analysis(basis, unper, per; ε_extra::Bool)
    data_length = min(length(unper.εs), length(per.εs))
    @assert unper.εs[1:data_length] ≈ per.εs[1:data_length]
    
    ρs_unper = unper.ρs
    vs_unper = unper.vs


    Δρ       = per.ρref - unper.ρref  
    Δρ_hm1   = norm_hm1(basis, Δρ)

    Qεs = Float64[]
    Rεs = Float64[]
    Sεs = Float64[]
    for i in 1:data_length
        (; εs, ρs, vs ) = per
        
        # Qε =  ‖ρ^ε - ρ̃^ε‖_H^-1 / ‖Δρ‖_H^-1
        Qε = norm_hm1(basis, ρs_unper[i] .- ρs[i]) / Δρ_hm1 
        # Rε = ε ‖v^ε - ṽ^ε‖_H^1 / ‖Δρ‖_H^-1    
        Rε = norm_h1(basis,  vs_unper[i] .- vs[i]) / Δρ_hm1 * εs[i]
        # Sε = ε ‖v^ε - ṽ^ε - J(Δρ/ε)‖_H^1 / ‖Δρ‖_H^-1
        Sε = norm_h1(basis,  vs_unper[i] .- vs[i] .- apply_J(basis, Δρ, εs[i])) / Δρ_hm1 * εs[i]
        
        push!(Qεs, Qε)
        push!(Rεs, Rε)
        push!(Sεs, Sε)
    end

    norms_ρ2orig_hm1 = [norm_hm1(basis, ρ  - per.ρorig)    for ρ  in per.ρs]
    norms_ρ2orig_l2  = [norm_l2(basis,  ρ  - per.ρorig)    for ρ  in per.ρs]

    (; Δρ_hm1, εs=unper.εs[1:data_length], Qεs, Rεs, Sεs, norms_ρ2orig_hm1, norms_ρ2orig_l2)
end
function compute_perturbation_analysis(unperturbed::AbstractString, perturbed::AbstractString; ε_extra=false)
    # unperturbed: File with the reference (unperturbed) inversion
    # perturbed:   File with the inversion

    @assert isfile(unperturbed)
    @assert is_inversion(unperturbed)
    ref_scfres = load_scfres(unperturbed; skip_hamiltonian=true);
    refres = jldopen(unperturbed) do jld
        (; vs    = jld["inversion_vs"],
           ρs    = jld["inversion_ρs"],
           εs    = jld["inversion_εs"],
           ρref  = jld["inversion_ρref"],
           ρorig = jld["inversion_ρorig"])
    end

    @assert isfile(perturbed)
    @assert is_inversion(perturbed)
    perturbres = jldopen(perturbed) do jld
        (; vs    = jld["inversion_vs"],
           ρs    = jld["inversion_ρs"],
           εs    = jld["inversion_εs"],
           ρref  = jld["inversion_ρref"],
           ρorig = jld["inversion_ρorig"])
    end

    bn, _ = splitext(perturbed)
    if ε_extra == true && isfile(bn * "_extra.jld2")
        println("   Using extra ε sequence")
        perturbres = jldopen(bn * "_extra.jld2") do jld
            εs = jld["inversion_εs"]
            ρs = jld["inversion_ρs"]
            vs = jld["inversion_vs"]
            εmask = (εs .< perturbres.εs[end])
            if any(εmask)
                return (;
                    εs    = [perturbres.εs; εs[εmask]],
                    ρs    = [perturbres.ρs; ρs[εmask]],
                    vs    = [perturbres.vs; vs[εmask]],
                    ρref  = perturbres.ρref,
                    ρorig = perturbres.ρorig)
            else
                return perturbres
            end
        end
    end
    compute_perturbation_analysis(ref_scfres.basis, refres, perturbres; ε_extra)
end

function extract_perturbation_analysis(unperturbed::AbstractString, perturbed::AbstractString; ε_extra=false)
    bn, _ = splitext(basename(perturbed))
    println(" ")
    println("Pertubation analysis for $(bn)")
    computed = compute_perturbation_analysis(unperturbed, perturbed; ε_extra)
    data = Dict(
        "Δρ_hm1"                    => computed.Δρ_hm1,
        "εs"                        => computed.εs,
        "Qεs"                       => computed.Qεs,
        "Rεs"                       => computed.Rεs,
        "Sεs"                       => computed.Sεs,
        "truncation_errors_ρ_hm1"   => computed.norms_ρ2orig_hm1,
        "truncation_errors_ρ_l2"    => computed.norms_ρ2orig_l2,
    )
    open(perturbfile(perturbed), "w") do fp
        JSON3.write(fp, data)
    end
    nothing
end

function main()
    #
    # Truncate on silicon Ecut 45 kgrid 10 Vxc
    #
    unperturbed_file = "results/inversion_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0.jld2"
    truncate_files = [
        "results/truncate$(trunc)_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0.jld2"
        for trunc in (10, 15, 20, 25, 30, 35)
    ]
    for trunc in truncate_files
        extract_perturbation_analysis(unperturbed_file, trunc; ε_extra=true)
    end

    unperturbed_file = "results/inversion_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0.jld2"
    truncate_files = [
        "results/truncate$(trunc)_silicon_Ecut_45_kgrid_10_alpha_1.0_beta_0.0_gamma_1.0.jld2"
        for trunc in (10, 15, 20, 25, 30, 35)
    ]
    for trunc in truncate_files
        extract_perturbation_analysis(unperturbed_file, trunc; ε_extra=true)
    end
end