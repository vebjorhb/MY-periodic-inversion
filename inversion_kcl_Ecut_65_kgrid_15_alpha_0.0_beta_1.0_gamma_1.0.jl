include("inversion.jl")
reference = "results/reference_kcl_Ecut_65_kgrid_15.jld2"
prefix, _ = splitext(@__FILE__)
prefix = joinpath("results", prefix)
εs = exp10.(0:-0.25:-7)
run_exact_inversion(prefix, reference; verbose=true, retry=2,
                    δ=1e-2, method=InversionVxcMod(0.0, 1.0, 1.0), εs, ρtol=5e-13)