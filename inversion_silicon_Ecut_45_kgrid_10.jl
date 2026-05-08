include("inversion.jl")
reference = "reference_silicon_Ecut_45_kgrid_10.jld2"
prefix, _ = splitext(@__FILE__)
εs = exp10.(0:-0.125:-7)
run_exact_inversion(prefix, reference; verbose=true,
                    δ=1e-2, method=InversionVxc(), εs, ρtol=5e-13)
