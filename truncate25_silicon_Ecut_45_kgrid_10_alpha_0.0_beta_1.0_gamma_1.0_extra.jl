include("inversion.jl")
reference = "results/reference_silicon_Ecut_45_kgrid_10.jld2"
prefix, _ = splitext(@__FILE__)

unperturbed = "results/inversion_silicon_Ecut_45_kgrid_10_alpha_0.0_beta_1.0_gamma_1.0.jld2"
verbose_unperturbed_data = jldopen(unperturbed) do jld
    (; vs=jld["inversion_vs"], ρs=jld["inversion_ρs"], εs=jld["inversion_εs"], ρref=jld["inversion_ρref"], )
end

εs = exp10.(-4.75:-0.25:-6)
run_inversion(prefix, reference, TruncateBasis(25); verbose=true, retry=0, 
                    verbose_unperturbed_data, δ=1e-2, ρtol=1e-10,
                    method=InversionVxcMod(0.0, 1.0, 1.0), εs)